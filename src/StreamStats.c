#include "StreamStats.h"
#include "Limelight-internal.h"

#include <string.h>

// Internal accumulator — not exposed in the header
typedef struct {
    PLT_MUTEX lock;

    // Jitter state — written only by the video receive thread (single producer,
    // no concurrent writer). Still guarded by lock for safe read in computeInterval.
    uint64_t lastFrameRecvTimeUs;
    uint64_t lastFramePtsUs;
    uint32_t jitterUs;   // RFC 3550 running EWMA estimate (µs)

    // Per-interval counters — incremented by the video thread, reset by the stats thread.
    uint32_t intervalDataPackets;
    uint32_t intervalParityPackets;
    uint32_t intervalLostPackets;   // pre-FEC holes (missingPackets)
    uint32_t intervalFramesDone;
    uint64_t intervalBytesApprox;   // (data + parity packets) * StreamConfig.packetSize

    // Latest frame-loss permille from the ControlStream frame-loss interval.
    // Written from connectionSawFrame() (control receive thread or video thread),
    // read in computeInterval(). Guarded by lock.
    uint16_t frameLossPermille;

    // Computed snapshot — written by the stats thread, read by any thread.
    STREAM_STAT_SNAPSHOT snapshot;
    bool snapshotValid;
} StreamStatAccum;

static StreamStatAccum g_accum;

void streamStatsInitialize(void) {
    memset(&g_accum, 0, sizeof(g_accum));
    PltCreateMutex(&g_accum.lock);
}

void streamStatsCleanup(void) {
    PltDeleteMutex(&g_accum.lock);
}

void streamStatsRecordFrame(uint64_t recvTimeUs, uint64_t ptsUs,
                             uint32_t dataPackets, uint32_t parityPackets,
                             uint32_t missingPackets) {
    PltLockMutex(&g_accum.lock);

    // RFC 3550 §6.4.1 inter-arrival jitter EWMA.
    // D(i) = |(recv[i] - recv[i-1]) - (pts[i] - pts[i-1])|
    // The constant clock offset between client and server cancels out in the
    // difference, so no clock synchronisation is needed.
    if (g_accum.lastFrameRecvTimeUs != 0) {
        int64_t dRecv = (int64_t)recvTimeUs - (int64_t)g_accum.lastFrameRecvTimeUs;
        int64_t dPts  = (int64_t)ptsUs      - (int64_t)g_accum.lastFramePtsUs;
        int64_t d     = dRecv - dPts;
        if (d < 0) d = -d;
        // jitter += (|d| - jitter) / 16
        g_accum.jitterUs = (uint32_t)((int64_t)g_accum.jitterUs +
                                      (d - (int64_t)g_accum.jitterUs) / 16);
    }
    g_accum.lastFrameRecvTimeUs = recvTimeUs;
    g_accum.lastFramePtsUs      = ptsUs;

    // Accumulate interval counters
    g_accum.intervalDataPackets   += dataPackets;
    g_accum.intervalParityPackets += parityPackets;
    g_accum.intervalLostPackets   += missingPackets;
    g_accum.intervalFramesDone++;

    // Bandwidth approximation: nominal packet size * all packets received this frame.
    // FEC parity is included since it is real network traffic.
    g_accum.intervalBytesApprox +=
        (uint64_t)(dataPackets + parityPackets) * (uint64_t)StreamConfig.packetSize;

    PltUnlockMutex(&g_accum.lock);
}

void streamStatsUpdateFrameLoss(uint16_t frameLossPermille) {
    PltLockMutex(&g_accum.lock);
    g_accum.frameLossPermille = frameLossPermille;
    PltUnlockMutex(&g_accum.lock);
}

void streamStatsComputeInterval(uint32_t rttMs, uint32_t rttVarianceMs,
                                 uint32_t intervalMs) {
    PltLockMutex(&g_accum.lock);

    // Snapshot the interval counters
    uint32_t dataPackets   = g_accum.intervalDataPackets;
    uint32_t parityPackets = g_accum.intervalParityPackets;
    uint32_t lostPackets   = g_accum.intervalLostPackets;
    uint32_t frames        = g_accum.intervalFramesDone;
    uint64_t bytes         = g_accum.intervalBytesApprox;
    uint32_t jitterUs      = g_accum.jitterUs;
    uint16_t frameLoss     = g_accum.frameLossPermille;

    // Reset interval counters for the next period
    g_accum.intervalDataPackets   = 0;
    g_accum.intervalParityPackets = 0;
    g_accum.intervalLostPackets   = 0;
    g_accum.intervalFramesDone    = 0;
    g_accum.intervalBytesApprox   = 0;

    // ---- Derive metrics ----

    // Bandwidth: bytes * 8 bits/byte / intervalMs ms = kbps
    uint32_t bandwidthKbps = (intervalMs > 0 && bytes > 0)
        ? (uint32_t)((bytes * 8) / intervalMs)
        : 0;

    // Pre-FEC packet loss permille
    uint16_t pktLossPermille = 0;
    uint32_t totalPkts = dataPackets + lostPackets;
    if (totalPkts > 0) {
        uint32_t p = (lostPackets * 1000) / totalPkts;
        pktLossPermille = (uint16_t)(p > 1000 ? 1000 : p);
    }

    // Clamp intervalLostPkts to uint16_t range
    uint16_t intervalLostClamped =
        (lostPackets > 0xFFFF) ? 0xFFFF : (uint16_t)lostPackets;

    // Write the snapshot
    g_accum.snapshot.rttMs              = rttMs;
    g_accum.snapshot.rttVarianceMs      = rttVarianceMs;
    g_accum.snapshot.jitterUs           = jitterUs;
    g_accum.snapshot.bandwidthKbps      = bandwidthKbps;
    g_accum.snapshot.pktLossPermille    = pktLossPermille;
    g_accum.snapshot.frameLossPermille  = frameLoss;
    g_accum.snapshot.intervalFrames     = frames;
    g_accum.snapshot.intervalDataPkts   = dataPackets;
    g_accum.snapshot.intervalLostPkts   = intervalLostClamped;
    g_accum.snapshotValid               = true;

    PltUnlockMutex(&g_accum.lock);
}

bool streamStatsGetSnapshot(STREAM_STAT_SNAPSHOT* snap) {
    if (snap == NULL) return false;

    PltLockMutex(&g_accum.lock);
    bool valid = g_accum.snapshotValid;
    if (valid) {
        *snap = g_accum.snapshot;
    }
    PltUnlockMutex(&g_accum.lock);

    return valid;
}
