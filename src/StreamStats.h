#pragma once

#include <stdint.h>
#include <stdbool.h>

// Full snapshot of computed stream stats.
// Written by lossStatsThread via streamStatsComputeInterval().
// Read from any thread via streamStatsGetSnapshot() or LiGetStreamStats().
typedef struct _STREAM_STAT_SNAPSHOT {
    uint32_t rttMs;              // ENet control-channel RTT (ms)
    uint32_t rttVarianceMs;      // ENet RTT variance (ms)
    uint32_t jitterUs;           // RFC 3550 inter-frame receive jitter (µs)
    uint32_t bandwidthKbps;      // incoming video + FEC bitrate (kbps)
    uint16_t pktLossPermille;    // pre-FEC packet loss, 0–1000
    uint16_t frameLossPermille;  // frame-level loss rate, 0–1000
    uint32_t intervalFrames;     // frames completed during this interval
    uint32_t intervalDataPkts;   // data packets received during this interval
    uint16_t intervalLostPkts;   // lost packets (pre-FEC) during this interval
} STREAM_STAT_SNAPSHOT;

// Called once from initializeControlStream() / destroyControlStream().
void streamStatsInitialize(void);
void streamStatsCleanup(void);

// Called from RtpVideoQueue when a frame is completed (video receive thread, hot path).
//   recvTimeUs    : PltGetMicroseconds() when the first RTP packet of the frame arrived
//   ptsUs         : presentationTimeUs of the frame's first packet (RTP ts in µs)
//   dataPackets   : bufferDataPackets
//   parityPackets : bufferParityPackets
//   missingPackets: holes in the sequence at frame-complete time (pre-FEC loss)
void streamStatsRecordFrame(uint64_t recvTimeUs, uint64_t ptsUs,
                             uint32_t dataPackets, uint32_t parityPackets,
                             uint32_t missingPackets);

// Called from ControlStream.c::connectionSawFrame() at the end of each frame-loss interval.
// frameLossPermille is the computed loss for that interval in the 0–1000 range.
void streamStatsUpdateFrameLoss(uint16_t frameLossPermille);

// Called from lossStatsThread before building the IDX_LOSS_STATS packet.
// Snapshots the current interval counters, computes all derived metrics, resets counters.
// rttMs / rttVarianceMs are supplied by the caller from ENet.
// intervalMs is the nominal sending interval (ms), used for bandwidth estimation.
void streamStatsComputeInterval(uint32_t rttMs, uint32_t rttVarianceMs,
                                 uint32_t intervalMs);

// Copy the latest computed snapshot into *snap.
// Returns false if streamStatsComputeInterval() has not been called yet.
// Safe to call from any thread.
bool streamStatsGetSnapshot(STREAM_STAT_SNAPSHOT* snap);
