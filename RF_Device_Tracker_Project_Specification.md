# Invisible Device Tracker using RF Fingerprinting

## Project Specification Document

**Project Type:** Undergraduate/Minor Project  
**Department:** Electronics and Communication Engineering  
**Implementation Level:** Student-Grade Practical

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Objective](#2-objective)
3. [Proposed Solution](#3-proposed-solution)
4. [System Architecture](#4-system-architecture)
5. [Hardware Requirements](#5-hardware-requirements)
6. [Software Requirements](#6-software-requirements)
7. [Methodology](#7-methodology)
8. [Expected Output](#8-expected-output)
9. [Applications](#9-applications)
10. [Challenges and Limitations](#10-challenges-and-limitations)
11. [Future Scope](#11-future-scope)
12. [Innovation](#12-innovation)

---

## 1. Problem Statement

### 1.1 The Need for Device Identification and Tracking

In modern environments, the ability to identify and track wireless devices serves critical purposes across security, logistics, healthcare, and industrial domains. Organizations need to know:

- **Who** is accessing their network or facility
- **Where** devices and assets are located at any given time
- **When** devices enter or leave designated areas

Traditional tracking methods either fail in indoor environments, require device cooperation, or can be easily circumvented.

### 1.2 Limitations of Existing Solutions

| Method | Working Principle | Major Limitations |
|--------|------------------|-------------------|
| **GPS Tracking** | Satellite trilateration | Fails indoors, high power consumption, requires clear sky view, expensive modules, add-on hardware required |
| **MAC Address** | Unique hardware identifier | Easily spoofed in software, requires active transmission, modern OS randomizes MAC for privacy |
| **Bluetooth Beacons** | Active Bluetooth transmission | Requires device Bluetooth to be ON, battery-dependent, short range (~10m), can be disabled |
| **WiFi-based RTLS** | WiFi signal analysis | Requires device to actively transmit, high infrastructure cost, affected by multipath fading |
| **RFID Tags** | Electromagnetic coupling | Short range (passive), requires close proximity reader, active tags need battery replacement |
| **Camera/Vision** | Visual recognition | Privacy concerns, requires line-of-sight, can be blocked, computationally expensive |

### 1.3 Real-World Gaps Addressed

| Gap | Description |
|-----|-------------|
| **Indoor Tracking** | No solution works reliably inside buildings where GPS fails |
| **Passive Operation** | Most methods require the tracked device to actively participate |
| **Hardware-Level Identity** | Software-based IDs (MAC) can be changed; hardware IDs cannot |
| **Low-Cost Implementation** | Commercial tracking systems cost thousands of dollars |
| **Battery Independence** | Methods requiring device cooperation drain battery life |

---

## 2. Objective

### 2.1 Primary Objectives

1. **Device Identification via RF Fingerprinting**
   - Extract unique hardware-level characteristics from wireless device signals
   - Train a machine learning model to recognize known devices
   - Achieve >85% identification accuracy under controlled conditions

2. **Basic Tracking via RSSI Monitoring**
   - Monitor Received Signal Strength Indicator (RSSI) over time
   - Detect changes in device proximity (near/far status)
   - Indicate movement direction based on signal variation patterns

### 2.2 Secondary Objectives

3. **Real-Time Processing Pipeline**
   - Process signals with minimal latency (<1 second)
   - Display identification results and signal strength in real-time

4. **Practical Student Implementation**
   - Use low-cost hardware (<$100 total)
   - Implement using Python with accessible libraries
   - Complete within one academic semester

### 2.3 Project Scope Boundaries

| Included | Not Included |
|----------|--------------|
| Single-device identification | Multi-device simultaneous tracking |
| RSSI-based proximity detection | Precise indoor positioning (TDoA/AoA) |
| Single SDR receiver | Multiple synchronized SDR nodes |
| Basic ML classification | Deep neural networks |
| 2D proximity status (near/far) | 3D location estimation |

---

## 3. Proposed Solution

### 3.1 RF Fingerprinting Concept

#### 3.1.1 Simple Explanation

Every wireless device—such as a smartphone, remote control, or IoT sensor—contains hardware components that are never perfectly identical. Tiny manufacturing variations create unique "electronic fingerprints" in how each device transmits signals. These fingerprints are:

- **Inherent**: Built into the hardware during manufacturing
- **Unique**: No two devices have exactly the same characteristics
- **Inalienable**: Cannot be changed without modifying the physical hardware
- **Passive**: Can be detected without any cooperation from the target device

#### 3.1.2 Technical Explanation

RF fingerprinting exploits **hardware impairments** in RF transceivers that arise from manufacturing tolerances:

```
Manufacturing Imperfection → Hardware Impairment → Unique Signal Characteristic
```

**Key Hardware Impairments:**

| Impairment | Origin | Signal Effect |
|------------|--------|---------------|
| **I/Q Imbalance** | Mismatch between I and Q mixer branches | Asymmetric constellation, rotated scatter plot |
| **Carrier Frequency Offset (CFO)** | Oscillator frequency mismatch | Constant phase rotation over time |
| **Phase Noise** | Oscillator jitter | Random phase fluctuations |
| **Amplitude Imbalance** | Different gain in I/Q paths | Elliptical I/Q scatter (not circular) |
| **DC Offset** | LO leakage in mixer | Off-center I/Q constellation |
| **Power Amplifier Non-linearity** | Compressed gain at high power | Harmonic distortion, spectral regrowth |

```
Device A: CFO = -847 Hz, I/Q Amplitude Imbalance = 0.023 dB, Phase Noise = -78 dBc/Hz
Device B: CFO = +1,203 Hz, I/Q Amplitude Imbalance = 0.041 dB, Phase Noise = -71 dBc/Hz
                                              ↓
                                    ML Classifier distinguishes
```

### 3.2 RSSI-Based Tracking Concept

#### 3.2.1 Simple Explanation

Radio signals lose strength as they travel through space. The farther a device is from the receiver, the weaker the signal. By monitoring how strong the received signal is, we can estimate whether the device is:

- **Near**: Strong signal (e.g., -40 to -60 dBm)
- **Medium distance**: Moderate signal (e.g., -60 to -80 dBm)
- **Far**: Weak signal (e.g., below -80 dBm)

#### 3.2.2 Technical Explanation

**Free-Space Path Loss Model:**

```
RSSI = P_tx + G_tx + G_rx - PL(d)

Where:
- P_tx = Transmit power (typically fixed)
- G_tx, G_rx = Antenna gains
- PL(d) = Path loss at distance d

Path Loss: PL(d) = 20*log10(4πd/λ)  [for free space]
         = 20*log10(f) + 20*log10(d) - 27.55  [in dB, with f in MHz, d in meters]
```

**For 433 MHz (ISM band):**
```
PL(dB) = 20*log10(433) + 20*log10(d) - 27.55
       ≈ 52.4 + 20*log10(d) - 27.55
       = 24.85 + 20*log10(d)
```

**RSSI to Distance Mapping:**

| RSSI (dBm) | Estimated Distance | Proximity Status |
|------------|-------------------|------------------|
| > -40 | < 0.5 m | Very close |
| -40 to -55 | 0.5 - 1 m | Near |
| -55 to -70 | 1 - 2 m | Medium |
| -70 to -85 | 2 - 5 m | Far |
| < -85 | > 5 m | Very far |

**Movement Detection Algorithm:**
```
RSSI_change = RSSI_current - RSSI_previous

If RSSI_change > +threshold:  "Device moving TOWARD"
If RSSI_change < -threshold: "Device moving AWAY"
Otherwise:                    "Device stationary"
```

### 3.3 Combined Identification + Tracking System

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   ┌───────────────┐      ┌───────────────────┐                     │
│   │  WIRELESS     │      │  "I am Device A" │                     │
│   │  DEVICE       │─────▶│  (Unique RF       │──▶ IDENTIFICATION   │
│   │  TRANSMITS    │      │   Fingerprint)    │                     │
│   └───────────────┘      └───────────────────┘                     │
│                                                                     │
│           │                                                         │
│           │ Signal Strength (RSSI)                                  │
│           ▼                                                         │
│   ┌───────────────┐      ┌───────────────────┐                     │
│   │  RSSI Value:   │─────▶│  Distance Estimate│──▶ TRACKING        │
│   │  -62 dBm      │      │  ~2 meters away    │    STATUS          │
│   └───────────────┘      └───────────────────┘                     │
│                                                                     │
│           │                                                         │
│           ▼                                                         │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │  OUTPUT: "Device A detected, approximately 2m away,         │  │
│   │           showing stable signal"                              │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. System Architecture

### 4.1 Complete System Block Diagram

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                      INVISIBLE DEVICE TRACKER SYSTEM                          ║
║                         Identification + Tracking                             ║
╚═══════════════════════════════════════════════════════════════════════════════╝

TARGET ENVIRONMENT
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                               │
│      ┌─────────────┐                                                          │
│      │ Device A    │ ╲                                                         │
│      │ (Smartphone)│  ╲   RF Signal                                           │
│      └─────────────┘   ╲                                                     │
│                          ╲                                                   │
│                           ╲  ╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱  │
│                           ╲╱                                                  │
│                          ╱                                                    │
│      ┌─────────────┐   ╱                                                       │
│      │ Device B    │╱                                                          │
│      │ (Remote)    │                                                           │
│      └─────────────┘                                                           │
│                                                                               │
│                         ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼                              │
│                    Radio Waves Propagating                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║  BLOCK 1: SIGNAL ACQUISITION                                                    ║
║  ┌───────────────────────────────────────────────────────────────────────────┐ ║
║  │                                                                           │ ║
║  │    ┌──────────────┐      ┌────────────────┐      ┌─────────────────┐   │ ║
║  │    │   ANTENNA    │──────▶│   RTL-SDR      │──────▶│  RAW I/Q DATA  │   │ ║
║  │    │  (433 MHz)   │       │   DONGLE       │       │  (Complex      │   │ ║
║  │    │   Dipole     │       │  R820T Tuner   │       │   Float 8-bit) │   │ ║
║  │    └──────────────┘      └────────────────┘      └────────┬────────┘   │ ║
║  │                                                              │            │ ║
║  │    Frequency Range: 500 kHz - 1.7 GHz                        │            │ ║
║  │    Sample Rate: 2.4 MSPS (max)                              │            │ ║
║  │    USB 2.0 Interface                                        │            │ ║
║  │                                                                           │ ║
║  └───────────────────────────────────────────────────────────────────────────┘ ║
╚═══════════════════════════════════════════════════════════════════════════════╝
                                    │
                                    │ USB
                                    ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║  BLOCK 2: SIGNAL PROCESSING (Python)                                            ║
║  ┌───────────────────────────────────────────────────────────────────────────┐ ║
║  │                                                                           │ ║
║  │    ┌─────────────────┐                                                 │ ║
║  │    │  BANDPASS       │      ┌─────────────────┐                        │ ║
║  │    │  FILTER         │─────▶│  DC OFFSET       │                        │ ║
║  │    │  (433 MHz ±    │       │  REMOVAL         │                        │ ║
║  │    │   2 MHz)       │       │                  │                        │ ║
║  │    └─────────────────┘       └────────┬────────┘                        │ ║
║  │                                        │                                  │ ║
║  │    ┌─────────────────┐                 │                                  │ ║
║  │    │  DOWN-          │◀────────────────┘                                  │ ║
║  │    │  SAMPLE         │                                                   │ ║
║  │    │  (2.4 MSPS →    │                                                   │ ║
║  │    │   1 MSPS)       │                                                   │ ║
║  │    └────────┬────────┘                                                   │ ║
║  │             │                                                             │ ║
║  │    ┌────────▼────────┐                                                    │ ║
║  │    │  AGC /          │                                                    │ ║
║  │    │  NORMALIZATION  │                                                    │ ║
║  │    └────────┬────────┘                                                    │ ║
║  │             │                                                              │ ║
║  │    ┌────────▼────────┐                                                    │ ║
║  │    │  PACKET         │───▶ Detection: Energy threshold                    │ ║
║  │    │  DETECTION      │───▶ Alignment: Find packet boundaries              │ ║
║  │    └────────┬────────┘───▶ RSSI: Calculate signal strength              │ ║
║  │             │                                                             │ ║
║  │             ▼                                                             │ ║
║  │    ┌─────────────────────────────────────────────────────────────────┐   │ ║
║  │    │  CLEAN I/Q SAMPLES + RSSI VALUE                                 │   │ ║
║  │    │  [I_samples[], Q_samples[], RSSI_dBm]                          │   │ ║
║  │    └─────────────────────────────────────────────────────────────────┘   │ ║
║  │                                                                           │ ║
║  └───────────────────────────────────────────────────────────────────────────┘ ║
╚═══════════════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║  BLOCK 3: FEATURE EXTRACTION                                                    ║
║  ┌───────────────────────────────────────────────────────────────────────────┐ ║
║  │                                                                           │ ║
║  │    INPUT: Clean I/Q Samples                                               │ ║
║  │    ┌─────────────────────────────────────────────────────────────────┐   │ ║
║  │    │                                                                  │   │ ║
║  │    │   ┌──────────────────────────────────────────────────────────┐  │   │ ║
║  │    │   │         TIME-DOMAIN FEATURES                            │  │   │ ║
║  │    │   │   • Mean amplitude                                       │  │   │ ║
║  │    │   │   • Variance of amplitude                                │  │   │ ║
║  │    │   │   • Peak-to-average ratio                                │  │   │ ║
║  │    │   │   • Zero-crossing rate                                   │  │   │ ║
║  │    │   └──────────────────────────────┬───────────────────────────┘  │   │ ║
║  │    │                                  │                             │   │ ║
║  │    │   ┌──────────────────────────────────────────────────────────┐  │   │ ║
║  │    │   │         FREQUENCY-DOMAIN FEATURES                        │  │   │ ║
║  │    │   │   • Power Spectral Density (PSD)                         │  │   │ ║
║  │    │   │   • Spectral centroid                                     │  │   │ ║
║  │    │   │   • Spectral flatness                                     │  │   │ ║
║  │    │   │   • Harmonic power ratios                                │  │   │ ║
║  │    │   └──────────────────────────────┬───────────────────────────┘  │   │ ║
║  │    │                                  │                             │   │ ║
║  │    │   ┌──────────────────────────────────────────────────────────┐  │   │ ║
║  │    │   │         I/Q DOMAIN FEATURES (RF Fingerprint)             │  │   │ ║
║  │    │   │                                                           │  │   │ ║
║  │    │   │   • Carrier Frequency Offset (CFO)                       │  │   │ ║
║  │    │   │   • I/Q Amplitude Imbalance                              │  │   │ ║
║  │    │   │   • I/Q Phase Imbalance                                   │  │   │ ║
║  │    │   │   • Circular variance (constellation spread)             │  │   │ ║
║  │    │   │   • Error Vector Magnitude (EVM)                          │  │   │ ║
║  │    │   │   • Phase noise PSD                                       │  │   │ ║
║  │    │   └───────────────────────────────────────────────────────────┘  │   │ ║
║  │    │                                                                  │   │ ║
║  │    └─────────────────────────────────────────────────────────────────┘   │ ║
║  │                                                                             │ ║
║  │    OUTPUT: Feature Vector                                                  │ ║
║  │    ┌─────────────────────────────────────────────────────────────────┐   │ ║
║  │    │  [CFO, I/Q_imb_A, I/Q_imb_P, Circ_Var, EVM, Phase_Noise, ...] │   │ ║
║  │    │   15-25 features per packet                                     │   │ ║
║  │    └─────────────────────────────────────────────────────────────────┘   │ ║
║  │                                                                           │ ║
║  └───────────────────────────────────────────────────────────────────────────┘ ║
╚═══════════════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║  BLOCK 4: MACHINE LEARNING CLASSIFICATION                                      ║
║  ┌───────────────────────────────────────────────────────────────────────────┐ ║
║  │                                                                           │ ║
║  │    ┌─────────────────────────────────────────────────────────────────┐   │ ║
║  │    │  FEATURE VECTOR                                                   │   │ ║
║  │    │  [0.023, -847, 0.041, -71, ...]                                   │   │ ║
║  │    └────────────────────────────────┬────────────────────────────────┘   │ ║
║  │                                     │                                     │ ║
║  │                                     ▼                                     │ ║
║  │    ┌─────────────────────────────────────────────────────────────────┐   │ ║
║  │    │                    ML CLASSIFIER                                │   │ ║
║  │    │  ┌─────────────────────────────────────────────────────────┐    │   │ ║
║  │    │  │                                                         │    │   │ ║
║  │    │  │    ┌─────────┐    ┌─────────┐    ┌─────────┐           │    │   │ ║
║  │    │  │    │   KNN   │    │   SVM   │    │Random   │           │    │   │ ║
║  │    │  │    │(Baseline)│   │(RBF)   │    │Forest   │           │    │   │ ║
║  │    │  │    └─────────┘    └─────────┘    └─────────┘           │    │   │ ║
║  │    │  │         │              │              │               │    │   │ ║
║  │    │  │         └──────────────┼──────────────┘               │    │   │ ║
║  │    │  │                          │                              │    │   │ ║
║  │    │  │                    ┌─────▼─────┐                        │    │   │ ║
║  │    │  │                    │  Best     │                        │    │   │ ║
║  │    │  │                    │  Model    │                        │    │   │ ║
║  │    │  │                    └─────┬─────┘                        │    │   │ ║
║  │    │  │                          │                               │    │   │ ║
║  │    │  └──────────────────────────┼───────────────────────────────┘    │   │ ║
║  │    │                                 │                                     │   │ ║
║  │    └─────────────────────────────────┼─────────────────────────────────┘   │ ║
║  │                                      │                                       │   ║
║  │                                      ▼                                       │   ║
║  │    ┌─────────────────────────────────────────────────────────────────┐   │ ║
║  │    │  CLASSIFICATION OUTPUT                                            │   │ ║
║  │    │                                                                  │   │ ║
║  │    │    Prediction: "Device_A"        Confidence: 94.2%             │   │ ║
║  │    │                                                                  │   │ ║
║  │    │    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │   │ ║
║  │    │    │ Device_A    │  │ Device_B    │  │  Unknown    │           │   │ ║
║  │    │    │   94.2%     │  │    4.1%     │  │    1.7%     │           │   │ ║
║  │    │    │   ████████  │  │   █        │  │   █         │           │   │ ║
║  │    │    └─────────────┘  └─────────────┘  └─────────────┘           │   │ ║
║  │    │                                                                  │   │ ║
║  │    └─────────────────────────────────────────────────────────────────┘   │ ║
║  │                                                                           │ ║
║  └───────────────────────────────────────────────────────────────────────────┘ ║
╚═══════════════════════════════════════════════════════════════════════════════╝
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║  BLOCK 5: RSSI-BASED TRACKING                                                   ║
║  ┌───────────────────────────────────────────────────────────────────────────┐ ║
║  │                                                                           │ ║
║  │    INPUT: RSSI Value + Historical RSSI Buffer                             │ ║
║  │    ┌─────────────────────────────────────────────────────────────────┐   │ ║
║  │    │                                                                  │   │ ║
║  │    │   ┌──────────────────────────────────────────────────────────┐  │   │ ║
║  │    │   │     RSSI HISTORY BUFFER                                  │  │   │ ║
║  │    │   │                                                          │  │   │ ║
║  │    │   │   t-10: -58 dBm  ──▶                                      │  │   │ ║
║  │    │   │   t-9:  -61 dBm  ──▶                                      │  │   │ ║
║  │    │   │   t-8:  -60 dBm  ──▶                                      │  │   │ ║
║  │    │   │   t-7:  -65 dBm  ──▶  [Moving AWAY]                       │  │   │ ║
║  │    │   │   t-6:  -68 dBm  ──▶                                      │  │   │ ║
║  │    │   │   t-5:  -70 dBm  ──▶                                      │  │   │ ║
║  │    │   │   t-4:  -72 dBm  ──▶                                      │  │   │ ║
║  │    │   │   t-3:  -74 dBm  ──▶                                      │  │   │ ║
║  │    │   │   t-2:  -73 dBm  ──▶                                      │  │   │ ║
║  │    │   │   t-1:  -75 dBm  ──▶                                      │  │   │ ║
║  │    │   │   t-0:  -76 dBm  ──▶  [Current]                           │  │   │ ║
║  │    │   │                                                          │  │   │ ║
║  │    │   └──────────────────────────────────────────────────────────┘  │   │ ║
║  │    │                                                                  │  │   │ ║
║  │    │   ┌──────────────────────────────────────────────────────────┐  │   │ ║
║  │    │   │     DISTANCE ESTIMATION                                   │  │   │ ║
║  │    │   │     RSSI = -76 dBm → Distance ≈ 3-5 meters               │  │   │ ║
║  │    │   │     Status: "FAR"                                         │  │   │ ║
║  │    │   └──────────────────────────────────────────────────────────┘  │   │ ║
║  │    │                                                                  │  │   │ ║
║  │    └─────────────────────────────────────────────────────────────────┘   │ ║
║  │                                                                             │ ║
║  │    ┌─────────────────────────────────────────────────────────────────┐   │ ║
║  │    │  TRACKING OUTPUT                                                │   │ ║
║  │    │                                                                  │   │ ║
║  │    │  • Proximity: "FAR"                                             │   │ ║
║  │    │  • Movement: "MOVING AWAY"                                     │   │ ║
║  │    │  • Est. Distance: 3-5 meters                                   │   │ ║
║  │    │  • Stability: "Decreasing signal (Δ = -3 dBm over window)"    │   │ ║
║  │    │                                                                  │   │ ║
║  │    └─────────────────────────────────────────────────────────────────┘   │ ║
║  │                                                                           │ ║
║  └───────────────────────────────────────────────────────────────────────────┘ ║
╚═══════════════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║  BLOCK 6: OUTPUT & USER INTERFACE                                               ║
║  ┌───────────────────────────────────────────────────────────────────────────┐ ║
║  │                                                                           │ ║
║  │    ┌─────────────────────────────────────────────────────────────────┐   │ ║
║  │    │                    TERMINAL / GUI OUTPUT                        │   │ ║
║  │    │                                                                  │   │ ║
║  │    │  ╔═══════════════════════════════════════════════════════════╗  │   │ ║
║  │    │  ║                                                           ║  │   │ ║
║  │    │  ║   DEVICE TRACKER v1.0                                      ║  │   │ ║
║  │    │  ║   ─────────────────────────────────────────────────       ║  │   │ ║
║  │    │  ║                                                           ║  │   │ ║
║  │    │  ║   DETECTED: Device_A                                       ║  │   │ ║
║  │    │  ║   CONFIDENCE: 94.2%                                         ║  │   │ ║
║  │    │  ║                                                           ║  │   │ ║
║  │    │  ║   ┌─────────────────────────────────────────────────┐    ║  │   │ ║
║  │    │  ║   │  SIGNAL STRENGTH: -76 dBm                        │    ║  │   │ ║
║  │    │  ║   │  ████████████░░░░░░░░░░░░░░░░░░░░░░░░░  (Moderate) │    ║  │   │ ║
║  │    │  ║   └─────────────────────────────────────────────────┘    ║  │   │ ║
║  │    │  ║                                                           ║  │   │ ║
║  │    │  ║   PROXIMITY: FAR                                         ║  │   │ ║
║  │    │  ║   MOVEMENT:  ──────────▶ (Moving Away)                    ║  │   │ ║
║  │    │  ║   EST. DISTANCE: 3-5 meters                               ║  │   │ ║
║  │    │  ║                                                           ║  │   │ ║
║  │    │  ║   TIMESTAMP: 2026-04-13 10:34:22                          ║  │   │ ║
║  │    │  ║                                                           ║  │   │ ║
║  │    │  ╚═══════════════════════════════════════════════════════════╝  │   │ ║
║  │    │                                                                  │   │ ║
║  │    └─────────────────────────────────────────────────────────────────┘   │ ║
║  │                                                                           │ ║
║  │    ┌─────────────────────────────────────────────────────────────────┐   │ ║
║  │    │                    SIGNAL STRENGTH PLOT                       │   │ ║
║  │    │                                                                  │   │ ║
║  │    │   -40 ─┬────────────────────────────────────                    │   │ ║
║  │    │        │                                                         │   │ ║
║  │    │        │                    ●●●●                                │   │ ║
║  │    │        │               ●●●●                                     │   │ ║
║  │    │   -60 ─┼──────────●●●●                                         │   │ ║
║  │    │        │     ●●●●                                               │   │ ║
║  │    │        │●●●●                                                     │   │ ║
║  │    │        │                                                         │   │ ║
║  │    │   -80 ─┼────────────────────────────────────────────────────    │   │ ║
║  │    │        │                                                         │   │ ║
║  │    │        └───────────────────────────────────────────────▶       │   │ ║
║  │    │        0s    5s    10s    15s    20s    25s    30s              │   │ ║
║  │    │                    TIME                                          │   │ ║
║  │    │                                                                  │   │ ║
║  │    │   Trend: Signal decreasing → Device moving away                │   │ ║
║  │    │                                                                  │   │ ║
║  │    └─────────────────────────────────────────────────────────────────┘   │ ║
║  │                                                                           │ ║
║  └───────────────────────────────────────────────────────────────────────────┘ ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

### 4.2 Data Flow Summary

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW DIAGRAM                                 │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│   ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐           │
│   │ RAW I/Q   │───▶│ PREPROCESS│───▶│  FEATURE  │───▶│    ML     │           │
│   │ (Complex) │    │           │    │ EXTRACT   │    │ CLASSIFIER│           │
│   └───────────┘    └───────────┘    └───────────┘    └─────┬─────┘           │
│                                                           │                  │
│                                                           ▼                  │
│   ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐           │
│   │  OUTPUT   │◀───│   GUI/    │◀───│  RSSI     │◀───│    ID     │           │
│   │  DISPLAY  │    │  TERMINAL │    │  ANALYSIS │    │ + LOCATION│           │
│   └───────────┘    └───────────┘    └───────────┘    └───────────┘           │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Hardware Requirements

### 5.1 Primary Components

| Component | Specification | Purpose | Quantity | Cost |
|-----------|--------------|---------|----------|------|
| **RTL-SDR Dongle** | RTL2838U + R820T2 tuner | RF signal capture | 1 | $25-35 |
| **Antenna** | 433 MHz dipole or telescopic (1090 MHz for ADS-B) | Receive RF signals | 1 | $10-15 |
| **USB Cable** | USB 2.0, 1-2 meters | Connect SDR to PC | 1 | $5 |
| **Laptop/Desktop** | 4GB+ RAM, USB 2.0, Python 3.8+ | Processing host | 1 | Existing |
| **USB Extension Cable** | Active USB 2.0 extension (optional) | Better antenna placement | 1 | $10 |

**Total Minimum Cost: ~$40-60**

### 5.2 Optional Enhancement Components

| Component | Purpose | Additional Cost |
|-----------|---------|-----------------|
| **Low-Noise Amplifier (LNA)** | Improve weak signal reception | $10-15 |
| **Better Antenna** | Vivaldi or discone for wider frequency | $30-50 |
| **USB 3.0 Hub (Powered)** | If connecting multiple SDRs | $20-25 |
| **Second RTL-SDR** | Synchronized dual-channel capture | $25-35 |

### 5.3 Target Frequency Bands for Student Projects

| Band | Frequency | Protocols/Devices | RTL-SDR Compatible |
|------|-----------|-------------------|-------------------|
| **ISM 433 MHz** | 433.05-434.79 MHz | Remote controls, IoT sensors, weather stations | Yes (native) |
| **ISM 915 MHz** | 902-928 MHz | LoRa, some IoT devices | Partial (needs upconverter) |
| **ADS-B 1090 MHz** | 1090 MHz | Aircraft transponders | Yes (native) |
| **ISM 2.4 GHz** | 2.4-2.5 GHz | WiFi, Bluetooth | Needs upconverter |
| **ISM 5.8 GHz** | 5.725-5.875 GHz | FPV drones, some WiFi | Needs upconverter |

**Recommended for Beginners:** 433 MHz ISM band
- No upconverter needed
- Many low-power devices to test with
- Good range (~100m outdoor)
- Simple signal formats

### 5.4 Hardware Setup Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HARDWARE SETUP                                    │
│                                                                             │
│                                                                             │
│    ┌──────────────┐                                                         │
│    │   TARGET     │                                                         │
│    │   DEVICE     │                                                         │
│    │  (Remote,    │     ═══════════════════════════════════════════════     │
│    │   IoT, etc.) │     ═══════ RF Signal ════════════════════════════     │
│    │              │     ═══════════════════════════════════════════════     │
│    └──────┬───────┘         ╲                                            │
│           │                  ╲                                           │
│           │                   ╲                                          │
│           │                    ╲                                         │
│           │                     ╲                                         │
│           │                      ╲╲                                       │
│           │                       ╲                                       │
│           │                        ╲                                      │
│           │                         ╲                                     │
│           │                          ╲                                    │
│           │                           ╲                                   │
│           │                            ▼                                  │
│           │                   ┌─────────────────┐                          │
│           │                   │     ANTENNA     │                          │
│           │                   │   (433 MHz      │                          │
│           │                   │    Dipole)      │                          │
│           │                   └────────┬────────┘                          │
│           │                            │                                   │
│           │                            │ Coax Cable                        │
│           │                            │                                   │
│           │                            ▼                                   │
│           │                   ┌─────────────────┐                          │
│           │                   │   RTL-SDR       │                          │
│           │                   │   DONGLE        │                          │
│           │                   │   ┌─────────┐   │                          │
│           │                   │   │USB Plug│   │                          │
│           │                   │   └────┬────┘   │                          │
│           │                   └────────┼────────┘                          │
│           │                            │ USB                               │
│           │                            │                                   │
│           │                            ▼                                   │
│           │                   ┌─────────────────┐                          │
│           │                   │   LAPTOP/PC     │                          │
│           │                   │                 │                          │
│           │                   │  Python +       │                          │
│           │                   │  SDR Software   │                          │
│           │                   │                 │                          │
│           │                   │  ┌───────────┐  │                          │
│           │                   │  │  Output   │  │                          │
│           │                   │  │  Display  │  │                          │
│           │                   │  └───────────┘  │                          │
│           │                   └─────────────────┘                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Software Requirements

### 6.1 Programming Language

| Language | Version | Purpose |
|----------|---------|---------|
| **Python** | 3.8+ | Primary development language |

### 6.2 Required Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| **numpy** | 1.20+ | Numerical computing, array operations |
| **scipy** | 1.7+ | Signal processing, filtering, FFT |
| **sklearn** | 1.0+ | Machine learning classification |
| **matplotlib** | 3.4+ | Signal visualization and plots |
| **pyrtlsdr** | (latest) | RTL-SDR hardware interface |
| **pandas** | 1.3+ | Data handling and storage |

### 6.3 Library Installation

```bash
# Core dependencies
pip install numpy scipy matplotlib pandas

# Machine learning
pip install scikit-learn

# RTL-SDR interface
pip install pyrtlsdr

# Optional: For better visualization
pip install pyqtgraph
```

### 6.4 Software Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SOFTWARE ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        MAIN APPLICATION                                │   │
│  │                         (tracker_main.py)                             │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                      │                                       │
│         ┌────────────────────────────┼────────────────────────────┐         │
│         │                            │                            │         │
│         ▼                            ▼                            ▼         │
│  ┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐  │
│  │  SIGNAL ACQUISI │        │  IDENTIFICATION │        │   TRACKING      │  │
│  │  TION MODULE    │        │  MODULE         │        │   MODULE        │  │
│  │                 │        │                 │        │                 │  │
│  │ ┌─────────────┐ │        │ ┌─────────────┐ │        │ ┌─────────────┐ │  │
│  │ │sdr_reader.py│ │        │ │features.py  │ │        │ │rssi_tracker │ │  │
│  │ │             │ │        │ │             │ │        │ │.py          │ │  │
│  │ │- init_sdr() │ │        │ │- extract_   │ │        │ │             │ │  │
│  │ │- read_samples│ │        │ │  features() │ │        │ │- init_track │ │  │
│  │ │- get_rssi() │ │        │ │- extract_cfo │ │        │ │- update_rssi │ │  │
│  │ └─────────────┘ │        │ │- extract_iq_ │ │        │ │- calc_status │ │  │
│  │                  │        │ │  imbalance  │ │        │ │- detect_move │ │  │
│  │ ┌─────────────┐ │        │ └──────┬──────┘ │        │ └──────┬──────┘ │  │
│  │ │signal_proc. │ │        │        │       │        │        │       │  │
│  │ │py           │ │        │ ┌──────▼──────┐ │        │ ┌──────▼──────┐ │  │
│  │ │             │ │        │ │classifier.py│ │        │ │tracker_ui.py│ │  │
│  │ │- filter()   │ │        │ │             │ │        │ │             │ │  │
│  │ │- normalize()│ │        │ │- train()    │ │        │ │- plot_rssi()│ │  │
│  │ │- detect_    │ │        │ │- predict()  │ │        │ │- show_status│ │  │
│  │ │  packet()   │ │        │ │- evaluate() │ │        │ └─────────────┘ │  │
│  │ └─────────────┘ │        │ └─────────────┘ │        │                 │  │
│  └────────┬─────────┘        └────────┬─────────┘        └─────────────────┘  │
│           │                          │                                       │
│           │         ┌─────────────────┴─────────────────┐                     │
│           │         │                                   │                     │
│           │         ▼                                   ▼                     │
│           │  ┌─────────────────┐              ┌─────────────────┐              │
│           │  │   MODEL FILES  │              │   DATA STORAGE │              │
│           │  │                 │              │                 │              │
│           │  │- model.pkl      │              │- train_data.csv │              │
│           │  │- scaler.pkl     │              │- device_profiles│              │
│           │  │- label_enc.     │              │  .json          │              │
│           │  │  .pkl           │              │                 │              │
│           │  └─────────────────┘              └─────────────────┘              │
│           │                                                                 │
│           └─────────────────────────────────────────────────────────────►   │
│                                    │                                          │
│                                    ▼                                          │
│                          ┌─────────────────┐                                  │
│                          │  OUTPUT LAYER   │                                  │
│                          │                 │                                  │
│                          │ • Console print │                                  │
│                          │ • Matplotlib    │                                  │
│                          │   real-time     │                                  │
│                          │   graphs        │                                  │
│                          │ • JSON log      │                                  │
│                          └─────────────────┘                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.5 File Structure

```
rf_device_tracker/
├── README.md
├── requirements.txt
├── tracker_main.py           # Main application entry point
├── config.py                # Configuration parameters
├── modules/
│   ├── __init__.py
│   ├── sdr_reader.py        # RTL-SDR interface
│   ├── signal_processor.py  # Preprocessing functions
│   ├── feature_extractor.py # RF fingerprint extraction
│   ├── classifier.py        # ML model training and prediction
│   └── rssi_tracker.py      # RSSI monitoring and tracking
├── utils/
│   ├── __init__.py
│   ├── filters.py           # Digital filter implementations
│   ├── math_utils.py        # Helper math functions
│   └── visualization.py     # Plotting utilities
├── models/
│   └── (trained model files saved here)
├── data/
│   ├── raw/                 # Raw captured signals
│   ├── processed/          # Processed features
│   └── dataset.csv          # Training dataset
├── training/
│   ├── collect_data.py      # Data collection script
│   ├── train_model.py       # Model training script
│   └── evaluate.py          # Model evaluation
└── output/
    ├── logs/                # Runtime logs
    └── plots/               # Generated plots
```

---

## 7. Methodology

### 7.1 Complete Working Process

#### Phase 1: Signal Acquisition

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PHASE 1: SIGNAL ACQUISITION                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Step 1.1: Hardware Connection                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  1. Connect RTL-SDR dongle to USB port                               │   │
│  │  2. Attach antenna to RTL-SDR SMA connector                           │   │
│  │  3. Verify device detection:                                          │   │
│  │     $ python -c "from pyrtlsdr import librtlsdr; print('OK')"        │   │
│  │  4. Test basic reception (e.g., FM radio or known signal)            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Step 1.2: SDR Configuration                                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Parameter          │ Value           │ Notes                        │   │
│  │  ───────────────────┼─────────────────┼────────────────────────────── │   │
│  │  Center Frequency   │ 433.92 MHz     │ Common ISM band              │   │
│  │  Sample Rate        │ 2.4 MSPS       │ Maximum for RTL-SDR          │   │
│  │  Gain Mode          │ Manual          │ Or 'auto' for testing        │   │
│  │  Manual Gain        │ 40-50 dB       │ Adjust based on signal       │   │
│  │  Bandwidth          │ 2 MHz          │ Default                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Step 1.3: Signal Capture Process                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │   Initialize SDR ──▶ Set Frequency ──▶ Set Sample Rate              │   │
│  │        │                      │                 │                    │   │
│  │        │                      ▼                 ▼                    │   │
│  │        │              ┌──────────────┐  ┌──────────────┐             │   │
│  │        │              │ Read I/Q     │  │ Start        │             │   │
│  │        │              │ Samples       │  │ Streaming    │             │   │
│  │        │              │ (1024-8192)   │  │ Thread       │             │   │
│  │        │              └──────┬───────┘  └──────┬───────┘             │   │
│  │        │                     │                   │                      │   │
│  │        │                     └─────────┬─────────┘                      │   │
│  │        │                               ▼                               │   │
│  │        │                      ┌──────────────┐                         │   │
│  │        │                      │ I/Q Buffer   │                         │   │
│  │        │                      │ [I, Q, I, Q] │                         │   │
│  │        │                      │ Complex:     │                         │   │
│  │        │                      │ [I+jQ, ...]  │                         │   │
│  │        │                      └──────────────┘                         │   │
│  │        │                                                            │   │
│  │        ▼                                                            │   │
│  │   ┌────────────────────────────────────────────────────────────┐    │   │
│  │   │  RAW I/Q DATA FORMAT:                                       │    │   │
│  │   │  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐    │    │   │
│  │   │  │ I0 │ Q0 │ I1 │ Q1 │ I2 │ Q2 │ I3 │ Q3 │ I4 │ Q4 │... │    │    │   │
│  │   │  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘    │    │   │
│  │   │  Data type: uint8 (0-255) or float32                      │    │   │
│  │   │  Sample rate: 2.4 million samples per second               │    │   │
│  │   └────────────────────────────────────────────────────────────┘    │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Phase 2: Preprocessing

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PHASE 2: PREPROCESSING                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Step 2.1: Convert to Complex Float                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │   uint8 [0, 255] ──▶ float32 [-1.0, +1.0]                           │   │
│  │                                                                       │   │
│  │   I_float = (I_uint8 / 128.0) - 1.0                                 │   │
│  │   Q_float = (Q_uint8 / 128.0) - 1.0                                 │   │
│  │                                                                       │   │
│  │   complex_samples = I_float + 1j * Q_float                          │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Step 2.2: DC Offset Removal                                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │   DC offset introduces a spurious carrier in the center              │   │
│  │                                                                       │   │
│  │   samples_corrected = samples - mean(samples)                      │   │
│  │                                                                       │   │
│  │   Before:                        After:                             │   │
│  │   ┌─────────────────┐          ┌─────────────────┐                 │   │
│  │   │        *        │          │   *   *         │                 │   │
│  │   │      *   *       │          │  *   *   *      │                 │   │
│  │   │    *   *   *     │          │   *   *   *     │                 │   │
│  │   │      *   *       │          │  *   *   *      │                 │   │
│  │   │        *        │          │   *   *         │                 │   │
│  │   │     (offset)    │          │  (centered)     │                 │   │
│  │   └─────────────────┘          └─────────────────┘                 │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Step 2.3: Bandpass Filtering                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │   Passband: 433.92 MHz ± 1 MHz                                       │   │
│  │                                                                       │   │
│  │   ┌─────────────────────────────────────────────────────────┐        │   │
│  │   │                                                         │        │   │
│  │   │        Passband                                         │        │   │
│  │   │   ───────────┐                         ┌──────────────        │   │
│  │   │               │                         │                      │   │
│  │   │               │      Stop      Stop     │                      │   │
│  │   │               ├───────────────┼─────────┤                      │   │
│  │   │               │               │         │                      │   │
│  │   │   432.92 MHz  │   433.92 MHz  │ 434.92  │ 435.92 MHz            │   │
│  │   │                                                         │        │   │
│  │   └─────────────────────────────────────────────────────────┘        │   │
│  │                                                                       │   │
│  │   Filter type: FIR (windowed sinc) or IIR (Butterworth)            │   │
│  │   Implementation: scipy.signal.firwin() or butter()                 │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Step 2.4: Packet Detection (Energy Detection)                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │   1. Calculate signal energy in sliding window:                       │   │
│  │      energy[n] = |samples[n-n_window:n]²|                           │   │
│  │                                                                       │   │
│  │   2. Compare to threshold:                                           │   │
│  │      if energy > threshold: PACKET DETECTED                          │   │
│  │                                                                       │   │
│  │   ┌───────────────────────────────────────────────────────────┐      │   │
│  │   │                                                           │      │   │
│  │   │  Energy                                                   │      │   │
│  │   │    ▲                                                      │      │   │
│  │   │    │         ╱╲                                            │      │   │
│  │   │    │        ╱  ╲        ╱╲                               │      │   │
│  │   │    │       ╱    ╲      ╱  ╲        ╱╲                    │      │   │
│  │   │    │      ╱      ╲    ╱    ╲      ╱  ╲                   │      │   │
│  │   │────┼─────╱────────╲──╱──────╲────╱────╲──────────────────┼──▶   │   │
│  │   │    │                          Threshold                     │      │   │
│  │   │    └────────────────────────────────────────────────────────       │   │
│  │   │    Time ──▶                                                │      │   │
│  │   │                                                           │      │   │
│  │   │   Detected Packets:  ●────────●        ●────────●        │      │   │
│  │   │                                                           │      │   │
│  │   └───────────────────────────────────────────────────────────┘      │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Step 2.5: RSSI Calculation                                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │   RSSI = 10 * log10( mean(|sample|²) ) in dB                         │   │
│  │                                                                       │   │
│  │   Or more commonly for RTL-SDR:                                      │   │
│  │   RSSI_dBm = (sum of squared magnitudes / N) in dB scale             │   │
│  │                                                                       │   │
│  │   Note: RTL-SDR doesn't provide calibrated RSSI. For relative        │   │
│  │   tracking, raw RSSI values are sufficient.                           │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Phase 3: Feature Extraction

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PHASE 3: FEATURE EXTRACTION                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Step 3.1: Time-Domain Features                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │   Feature 1: Mean Amplitude                                          │   │
│  │   ┌───────────────────────────────────────────────────────────────┐   │   │
│  │   │ mean_amp = mean(|samples|)                                   │   │   │
│  │   └───────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  │   Feature 2: Variance of Amplitude                                    │   │
│  │   ┌───────────────────────────────────────────────────────────────┐   │   │
│  │   │ var_amp = var(|samples|)                                     │   │   │
│  │   └───────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  │   Feature 3: Peak-to-Average Power Ratio (PAPR)                      │   │
│  │   ┌───────────────────────────────────────────────────────────────┐   │   │
│  │   │ peak = max(|samples|)                                         │   │   │
│  │   │ avg = mean(|samples|)                                          │   │   │
│  │   │ papr = peak / avg                                              │   │   │
│  │   └───────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Step 3.2: Frequency-Domain Features                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │   1. Compute FFT of samples:                                         │   │
│  │      spectrum = fft(samples)                                         │   │
│  │      psd = |spectrum|²  (Power Spectral Density)                     │   │
│  │                                                                       │   │
│  │   Feature 4: Spectral Centroid                                       │   │
│  │   ┌───────────────────────────────────────────────────────────────┐   │   │
│  │   │ centroid = sum(frequency * psd) / sum(psd)                    │   │   │
│  │   └───────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  │   Feature 5: Spectral Flatness (ratio of geometric to arithmetic    │   │
│  │              mean of PSD - distinguishes noise from signals)          │   │
│  │   ┌───────────────────────────────────────────────────────────────┐   │   │
│  │   │ flatness = (Π PSD[i])^(1/N) / (sum(PSD)/N)                    │   │   │
│  │   └───────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  │   Feature 6: Band Power (total power in signal band)                 │   │
│  │   ┌───────────────────────────────────────────────────────────────┐   │   │
│  │   │ band_power = sum(psd[passband_indices])                       │   │   │
│  │   └───────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Step 3.3: I/Q Domain Features (RF Fingerprint)                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │   Feature 7: Carrier Frequency Offset (CFO)                        │   │
│  │   ┌───────────────────────────────────────────────────────────────┐   │   │
│  │   │ Method: Phase difference over time                            │   │   │
│  │   │                                                           │   │   │
│  │   │ delta_phi = angle(samples[n] * conj(samples[n-1]))          │   │   │
│  │   │ CFO_Hz = delta_phi * sample_rate / (2*pi)                   │   │   │
│  │   │ CFO = mean(delta_phi) * sample_rate / (2*pi)                 │   │   │
│  │   │                                                               │   │   │
│  │   │ Typical values: -5000 Hz to +5000 Hz                         │   │   │
│  │   └───────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  │   Feature 8: I/Q Amplitude Imbalance                                │   │
│  │   ┌───────────────────────────────────────────────────────────────┐   │   │
│  │   │ I_rms = sqrt(mean(I²))                                        │   │   │
│  │   │ Q_rms = sqrt(mean(Q²))                                        │   │   │
│  │   │ amp_imbalance = I_rms / Q_rms  (or vice versa)                │   │   │
│  │   │                                                               │   │   │
│  │   │ Typical values: 0.9 to 1.1 (ratio)                           │   │   │
│  │   └───────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  │   Feature 9: I/Q Phase Imbalance                                     │   │
│  │   ┌───────────────────────────────────────────────────────────────┐   │   │
│  │   │ ideal_phase = 90°                                             │   │   │
│  │   │ actual_phase = mean(atan2(Q, I)) * (180/pi)                   │   │   │
│  │   │ phase_imbalance = actual_phase - ideal_phase                  │   │   │
│  │   │                                                               │   │   │
│  │   │ Typical values: -5° to +5°                                    │   │   │
│  │   └───────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  │   Feature 10: Circular Variance (constellation tightness)           │   │
│  │   ┌───────────────────────────────────────────────────────────────┐   │   │
│  │   │ I/Q Scatter: ideal = circle, actual deviations = fingerprint │   │   │
│  │   │                                                               │   │   │
│  │   │ radius = sqrt(I² + Q²)                                        │   │   │
│  │   │ circ_variance = var(radius) / mean(radius)²                   │   │   │
│  │   │                                                               │   │   │
│  │   │ Lower value = tighter circle = different fingerprint         │   │   │
│  │   └───────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  │   Feature 11: Error Vector Magnitude (EVM)                           │   │
│  │   ┌───────────────────────────────────────────────────────────────┐   │   │
│  │   │ Reference: ideal signal on unit circle                       │   │   │
│  │   │ Error = actual_sample - reference_sample                      │   │   │
│  │   │ EVM = sqrt(mean(|error|²)) / sqrt(mean(|reference|²)) * 100%  │   │   │
│  │   │                                                               │   │   │
│  │   │ Typical values: 1% to 10%                                     │   │   │
│  │   └───────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Step 3.4: Feature Vector Assembly                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │ FEATURE_VECTOR = [                                          │   │   │
│  │   │     mean_amp,           # 1                                  │   │   │
│  │   │     var_amp,            # 2                                  │   │   │
│  │   │     papr,               # 3                                  │   │   │
│  │   │     spectral_centroid,  # 4                                  │   │   │
│  │   │     spectral_flatness,  # 5                                  │   │   │
│  │   │     band_power,         # 6                                  │   │   │
│  │   │     cfo,                # 7 (RF fingerprint)               │   │   │
│  │   │     iq_amp_imbalance,   # 8 (RF fingerprint)               │   │   │
│  │   │     iq_phase_imbalance, # 9 (RF fingerprint)              │   │   │
│  │   │     circular_variance,  # 10 (RF fingerprint)              │   │   │
│  │   │     evm,                # 11 (RF fingerprint)                │   │   │
│  │   │     ... (additional features as needed)                      │   │   │
│  │   │   ]                                                        │   │   │
│  │   │                                                             │   │   │
│  │   │ Shape: (11,) for each packet                                │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Phase 4: Model Training

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PHASE 4: MODEL TRAINING                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Step 4.1: Dataset Collection                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │   1. For each known device (A, B, C, etc.):                          │   │
│  │      a. Place device at fixed location                              │   │
│  │      b. Activate device to transmit                                 │   │
│  │      c. Capture 500-1000 packets                                    │   │
│  │      d. Extract features from each packet                           │   │
│  │      e. Label with device ID                                        │   │
│  │                                                                       │   │
│  │   2. Include negative samples:                                      │   │
│  │      a. Capture noise (no device transmitting)                     │   │
│  │      b. Label as "noise" or "unknown"                               │   │
│  │                                                                       │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │  DATASET FORMAT (CSV):                                       │   │   │
│  │   │                                                              │   │   │
│  │   │  packet_id,mean_amp,var_amp,papr,cfo,iq_imb,...,device_id   │   │   │
│  │   │  1          ,0.523   ,0.042  ,3.2 ,847  ,0.023  ,...,A        │   │   │
│  │   │  2          ,0.518   ,0.038  ,3.1 ,852  ,0.025  ,...,A        │   │   │
│  │   │  3          ,0.234   ,0.089  ,4.5 ,-1203 ,0.041  ,...,B        │   │   │
│  │   │  4          ,0.231   ,0.092  ,4.4 ,-1198 ,0.039  ,...,B        │   │   │
│  │   │  ...                                                            │   │   │
│  │   │                                                              │   │   │
│  │   │  Total samples: ~3000-5000 (500-1000 per device + noise)      │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Step 4.2: Feature Normalization                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │   Use StandardScaler to normalize features:                        │   │
│  │                                                                       │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │                                                              │   │   │
│  │   │   X_train_normalized = (X_train - mean) / std              │   │   │
│  │   │   X_test_normalized  = (X_test  - mean) / std               │   │   │
│  │   │                                                              │   │   │
│  │   │   Fit on training data ONLY, apply to test data             │   │   │
│  │   │                                                              │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Step 4.3: Model Selection and Training                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │   Option A: Random Forest (Recommended for beginners)              │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │  from sklearn.ensemble import RandomForestClassifier         │   │   │
│  │   │                                                              │   │   │
│  │   │  model = RandomForestClassifier(                            │   │   │
│  │   │      n_estimators=100,                                        │   │   │
│  │   │      max_depth=10,                                            │   │   │
│  │   │      random_state=42                                         │   │   │
│  │   │  )                                                            │   │   │
│  │   │                                                              │   │   │
│  │   │  model.fit(X_train, y_train)                                   │   │   │
│  │   │  accuracy = model.score(X_test, y_test)                       │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  │   Option B: Support Vector Machine (SVM)                            │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │  from sklearn.svm import SVC                                  │   │   │
│  │   │                                                              │   │   │
│  │   │  model = SVC(kernel='rbf', C=10, gamma='scale')              │   │   │
│  │   │  model.fit(X_train, y_train)                                   │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  │   Option C: K-Nearest Neighbors (KNN) - Baseline                    │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │  from sklearn.neighbors import KNeighborsClassifier          │   │   │
│  │   │                                                              │   │   │
│  │   │  model = KNeighborsClassifier(n_neighbors=5)                 │   │   │
│  │   │  model.fit(X_train, y_train)                                  │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Step 4.4: Cross-Validation and Evaluation                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │  from sklearn.model_selection import cross_val_score        │   │   │
│  │   │                                                              │   │   │
│  │   │  # 5-fold cross-validation                                   │   │   │
│  │   │  scores = cross_val_score(model, X, y, cv=5)                 │   │   │
│  │   │  print(f"CV Accuracy: {scores.mean():.2%} ± {scores.std():.2%}") │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  │   Evaluation Metrics:                                               │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │  • Accuracy: Overall correct predictions                     │   │   │
│  │   │  • Precision: True positives / (True pos + False pos)       │   │   │
│  │   │  • Recall: True positives / (True pos + False neg)           │   │   │
│  │   │  • F1-Score: Harmonic mean of precision and recall            │   │   │
│  │   │  • Confusion Matrix: Per-class performance                    │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Step 4.5: Save Model                                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │  import joblib                                               │   │   │
│  │   │                                                              │   │   │
│  │   │  # Save model                                                │   │   │
│  │   │  joblib.dump(model, 'models/rf_classifier.pkl')              │   │   │
│  │   │  joblib.dump(scaler, 'models/scaler.pkl')                   │   │   │
│  │   │                                                              │   │   │
│  │   │  # Load model later                                          │   │   │
│  │   │  model = joblib.load('models/rf_classifier.pkl')             │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Phase 5: Device Classification (Real-Time)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 5: DEVICE CLASSIFICATION                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Real-Time Classification Pipeline:                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │   ┌────────────┐                                                      │   │
│  │   │ RAW I/Q   │                                                      │   │
│  │   │ SAMPLES   │                                                      │   │
│  │   └─────┬──────┘                                                      │   │
│  │         │                                                              │   │
│  │         ▼                                                              │   │
│  │   ┌────────────┐                                                      │   │
│  │   │ PREPROCESS │                                                      │   │
│  │   │  - Convert │                                                      │   │
│  │   │  - Filter  │                                                      │   │
│  │   │  - Detect  │                                                      │   │
│  │   └─────┬──────┘                                                      │   │
│  │         │                                                              │   │
│  │         ▼                                                              │   │
│  │   ┌────────────┐                                                      │   │
│  │   │ EXTRACT    │                                                      │   │
│  │   │ FEATURES   │  ──▶  feature_vector [11 values]                     │   │
│  │   │ (CFO,I/Q) │                                                      │   │
│  │   └─────┬──────┘                                                      │   │
│  │         │                                                              │   │
│  │         ▼                                                              │   │
│  │   ┌────────────┐                                                      │   │
│  │   │ NORMALIZE  │                                                      │   │
│  │   │ (scaler)   │  ──▶  normalized_vector                              │   │
│  │   └─────┬──────┘                                                      │   │
│  │         │                                                              │   │
│  │         ▼                                                              │   │
│  │   ┌────────────┐                                                      │   │
│  │   │ PREDICT    │                                                      │   │
│  │   │ (model)    │──▶  ┌────────────────────────────────────────────┐  │   │
│  │   └─────┬──────┘     │  prediction: "Device_A"                     │  │   │
│  │         │             │  confidence: [0.942, 0.041, 0.017]        │  │   │
│  │         │             │  probabilities: Device_A=94.2%, B=4.1%   │  │   │
│  │         │             └────────────────────────────────────────────┘  │   │
│  │         │                                                              │   │
│  │         ▼                                                              │   │
│  │   ┌────────────┐                                                      │   │
│  │   │ THRESHOLD │  If max_confidence < 0.7: Flag as "Unknown"           │   │
│  │   │ CHECK     │                                                      │   │
│  │   └────────────┘                                                      │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Phase 6: RSSI-Based Tracking

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PHASE 6: RSSI-BASED TRACKING                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Step 6.1: Initialize RSSI Tracker for Each Device                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │   class RSSTracker:                                                  │   │
│  │       def __init__(self, device_id, buffer_size=50):                 │   │
│  │           self.device_id = device_id                                 │   │
│  │           self.history = []  # Circular buffer of RSSI values       │   │
│  │           self.buffer_size = buffer_size                            │   │
│  │           self.timestamps = []                                      │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Step 6.2: Update with New Measurement                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │   def update(self, rssi_dbm, timestamp):                              │   │
│  │       self.history.append(rssi_dbm)                                  │   │
│  │       self.timestamps.append(timestamp)                              │   │
│  │       if len(self.history) > self.buffer_size:                      │   │
│  │           self.history.pop(0)  # Remove oldest                       │   │
│  │           self.timestamps.pop(0)                                    │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Step 6.3: Calculate Proximity Status                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │   ┌───────────────────────────────────────────────────────────────┐   │   │
│  │   │  def get_proximity(self, current_rssi):                      │   │   │
│  │   │                                                               │   │   │
│  │   │      if current_rssi > -50:                                   │   │   │
│  │   │          return "VERY_NEAR"  # < 0.5m                        │   │   │
│  │   │      elif current_rssi > -65:                                 │   │   │
│  │   │          return "NEAR"      # 0.5 - 2m                       │   │   │
│  │   │      elif current_rssi > -80:                                 │   │   │
│  │   │          return "MEDIUM"    # 2 - 5m                        │   │   │
│  │   │      else:                                                     │   │   │
│  │   │          return "FAR"       # > 5m                            │   │   │
│  │   │                                                               │   │   │
│  │   │      # Note: Actual distance depends on environment and        │   │   │
│  │   │      # transmit power of target device                        │   │   │
│  │   └───────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Step 6.4: Detect Movement Direction                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │   ┌───────────────────────────────────────────────────────────────┐   │   │
│  │   │  def detect_movement(self, window=10, threshold=3.0):        │   │   │
│  │   │                                                               │   │   │
│  │   │      if len(self.history) < window:                           │   │   │
│  │   │          return "INSUFFICIENT_DATA"                           │   │   │
│  │   │                                                               │   │   │
│  │   │      # Calculate trend using linear regression               │   │   │
│  │   │      recent = self.history[-window:]                         │   │   │
│  │   │      x = range(len(recent))                                   │   │   │
│  │   │      slope, intercept = linear_regression(x, recent)           │   │   │
│  │   │                                                               │   │   │
│  │   │      # slope > 0: signal increasing (moving closer)          │   │   │
│  │   │      # slope < 0: signal decreasing (moving away)             │   │   │
│  │   │                                                               │   │   │
│  │   │      if slope > threshold:                                    │   │   │
│  │   │          return "MOVING_TOWARD"                               │   │   │
│  │   │      elif slope < -threshold:                                 │   │   │
│  │   │          return "MOVING_AWAY"                                 │   │   │
│  │   │      else:                                                     │   │   │
│  │   │          return "STATIONARY"                                  │   │   │
│  │   │                                                               │   │   │
│  │   └───────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  │   Movement Detection Visualization:                                    │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │                                                             │   │   │
│  │   │   MOVING TOWARD:          MOVING AWAY:        STATIONARY:  │   │   │
│  │   │       ▲                        ▼                    ────    │   │   │
│  │   │      ╱╲                       ╱╲                   ╱╱╱╱╱╱    │   │   │
│  │   │     ╱  ╲                     ╱  ╲                ╱╱╱╱╱╱    │   │   │
│  │   │    ╱    ╲                   ╱    ╲             ────────   │   │   │
│  │   │   ╱      ╲                 ╱      ╲            ╲╲╲╲╲╲╲    │   │   │
│  │   │  ────────               ────────              ╲╲╲╲╲╲╲    │   │   │
│  │   │ Signal ↑               Signal ↓               Signal ~    │   │   │
│  │   │                                                             │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Step 6.5: Estimate Distance (Optional)                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │   Using path loss model:                                              │   │
│  │   ┌───────────────────────────────────────────────────────────────┐   │   │
│  │   │                                                               │   │   │
│  │   │   RSSI = RSSI₀ - 10*n*log₁₀(d/d₀)                             │   │   │
│  │   │                                                               │   │   │
│  │   │   d = d₀ * 10^((RSSI₀ - RSSI) / (10*n))                      │   │   │
│  │   │                                                               │   │   │
│  │   │   Where:                                                       │   │   │
│  │   │     RSSI₀ = RSSI at reference distance (typically 1m)        │   │   │
│  │   │     n = path loss exponent (2 for free space, 3-4 indoors)  │   │   │
│  │   │     d₀ = reference distance (typically 1m)                    │   │   │
│  │   │                                                               │   │   │
│  │   │   Example: RSSI₀ = -50 dBm, n = 3, RSSI = -70 dBm            │   │   │
│  │   │   d = 1 * 10^(((-50) - (-70)) / 30) = 1 * 10^(20/30) ≈ 2.15m  │   │   │
│  │   │                                                               │   │   │
│  │   └───────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Complete Algorithm Flow

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                           MAIN EXECUTION LOOP                                  │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   INITIALIZATION                                                                │
│   ├── Load trained ML model                                                    │
│   ├── Load scaler                                                              │
│   ├── Initialize SDR                                                           │
│   ├── Create RSSI tracker dictionary: {device_id: tracker}                     │
│   └── Start streaming                                                          │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐  │
│   │                         CONTINUOUS LOOP                                  │  │
│   │                                                                          │  │
│   │   ┌────────────────┐                                                     │  │
│   │   │ Read I/Q       │                                                     │  │
│   │   │ Samples (1024)│                                                     │  │
│   │   └───────┬────────┘                                                     │  │
│   │           │                                                               │  │
│   │           ▼                                                               │  │
│   │   ┌────────────────┐     No    ┌────────────────┐                        │  │
│   │   │ Signal Above  │──────────▶│ Discard / Wait │                        │  │
│   │   │ Threshold?     │          │ Continue Loop  │                        │  │
│   │   └───────┬────────┘          └────────────────┘                        │  │
│   │           │ Yes                                                             │  │
│   │           ▼                                                                 │  │
│   │   ┌────────────────┐                                                     │  │
│   │   │ Extract Packet │───▶ Collect ~2048 samples for analysis             │  │
│   │   └───────┬────────┘                                                     │  │
│   │           │                                                                 │  │
│   │           ▼                                                                 │  │
│   │   ┌────────────────┐                                                     │  │
│   │   │ Calculate RSSI │───▶ Save RSSI value                                 │  │
│   │   └───────┬────────┘                                                     │  │
│   │           │                                                                 │  │
│   │           ▼                                                                 │  │
│   │   ┌────────────────┐                                                     │  │
│   │   │ Extract        │                                                     │  │
│   │   │ Features       │───▶ CFO, I/Q imbalance, etc.                        │  │
│   │   │ (11 features)  │                                                     │  │
│   │   └───────┬────────┘                                                     │  │
│   │           │                                                                 │  │
│   │           ▼                                                                 │  │
│   │   ┌────────────────┐                                                     │  │
│   │   │ Normalize      │───▶ Apply trained scaler                           │  │
│   │   └───────┬────────┘                                                     │  │
│   │           │                                                                 │  │
│   │           ▼                                                                 │  │
│   │   ┌────────────────┐                                                     │  │
│   │   │ ML Prediction  │───▶ Device ID + Confidence                          │  │
│   │   │                 │                                                     │  │
│   │   └───────┬────────┘                                                     │  │
│   │           │                                                                 │  │
│   │           ├─────────────────────────────────┐                            │  │
│   │           │                                 │                            │  │
│   │           ▼                                 ▼                            │  │
│   │   ┌────────────────┐             ┌────────────────┐                     │  │
│   │   │ Known Device?  │             │ Unknown Device │                     │  │
│   │   │ (Conf > 70%)   │             │ (Conf < 70%)   │                     │  │
│   │   └───────┬────────┘             └───────┬────────┘                     │  │
│   │           │                               │                              │  │
│   │           ▼                               ▼                              │  │
│   │   ┌────────────────┐             ┌────────────────┐                     │  │
│   │   │ Update RSSI    │             │ Log as Unknown │                     │  │
│   │   │ Tracker        │             │                │                     │  │
│   │   │ for this ID    │             │ Alert (if      │                     │  │
│   │   └───────┬────────┘             │  enabled)      │                     │  │
│   │           │                      └────────────────┘                     │  │
│   │           │                                                                 │  │
│   │           ▼                                                                 │  │
│   │   ┌────────────────┐                                                     │  │
│   │   │ Get Proximity  │───▶ VERY_NEAR / NEAR / MEDIUM / FAR                │  │
│   │   └───────┬────────┘                                                     │  │
│   │           │                                                                 │  │
│   │           ▼                                                                 │  │
│   │   ┌────────────────┐                                                     │  │
│   │   │ Detect         │───▶ MOVING_TOWARD / MOVING_AWAY / STATIONARY        │  │
│   │   │ Movement       │                                                     │  │
│   │   └───────┬────────┘                                                     │  │
│   │           │                                                                 │  │
│   │           ▼                                                                 │  │
│   │   ┌──────────────────────────────────────────────────────────────────┐   │  │
│   │   │ OUTPUT DISPLAY                                                    │   │  │
│   │   │                                                                   │   │  │
│   │   │ ════════════════════════════════════════════════════════════   │   │  │
│   │   │ Device: Device_A  |  Confidence: 94.2%                         │   │  │
│   │   │ RSSI: -62 dBm      |  Proximity: NEAR                            │   │  │
│   │   │ Movement: → STATIONARY                                          │   │  │
│   │   │ ════════════════════════════════════════════════════════════   │   │  │
│   │   │                                                                   │   │  │
│   │   │ [RSSI Plot showing last 60 seconds]                            │   │  │
│   │   │ ████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   │   │  │
│   │   │                                                                   │   │  │
│   │   └──────────────────────────────────────────────────────────────────┘   │  │
│   │                                                                          │  │
│   └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│   LOOP CONTINUES...                                                            │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Expected Output

### 8.1 Terminal Output Format

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                   INVISIBLE DEVICE TRACKER v1.0                               ║
║                   RF Fingerprinting + RSSI Tracking                            ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║  [2026-04-13 10:34:22] System initialized                                     ║
║  [2026-04-13 10:34:22] Model loaded: rf_classifier.pkl                        ║
║  [2026-04-13 10:34:22] SDR center frequency: 433.920 MHz                      ║
║  [2026-04-13 10:34:22] Sample rate: 2.400 MHz                                 ║
║  [2026-04-13 10:34:22] Tracking 3 known devices                               ║
║  ────────────────────────────────────────────────────────────────────────────  ║
║                                                                                ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │                                                                         │  ║
║  │  ┌─────────────────────────────────────────────────────────────────────┐│  ║
║  │  │  ● DEVICE_A (Remote Control)                                       ││  ║
║  │  │  ──────────────────────────────────────────────────────────────── ││  ║
║  │  │  STATUS:     ACTIVE            CONFIDENCE:  94.2%                 ││  ║
║  │  │  RSSI:       -62 dBm  ████████████████░░░░░░░░░░░░  (Moderate)    ││  ║
║  │  │  PROXIMITY:  NEAR        (~1-2 meters estimated)                  ││  ║
║  │  │  MOVEMENT:   STATIONARY  ─────────────────────────                ││  ║
║  │  │  LAST SEEN:  0.3 seconds ago                                      ││  ║
║  │  │                                                                     ││  ║
║  │  └─────────────────────────────────────────────────────────────────────┘│  ║
║  │                                                                         │  ║
║  │  ┌─────────────────────────────────────────────────────────────────────┐│  ║
║  │  │  ○ DEVICE_B (Temperature Sensor)                                   ││  ║
║  │  │  ──────────────────────────────────────────────────────────────── ││  ║
║  │  │  STATUS:     ACTIVE            CONFIDENCE:  91.8%                 ││  ║
║  │  │  RSSI:       -78 dBm  ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   ││  ║
║  │  │  PROXIMITY:  MEDIUM       (~3-4 meters estimated)                ││  ║
║  │  │  MOVEMENT:   MOVING_AWAY  ──────────────────────▶                 ││  ║
║  │  │  LAST SEEN:  1.2 seconds ago                                     ││  ║
║  │  │                                                                     ││  ║
║  │  └─────────────────────────────────────────────────────────────────────┘│  ║
║  │                                                                         │  ║
║  │  ┌─────────────────────────────────────────────────────────────────────┐│  ║
║  │  │  ⚠ UNKNOWN DEVICE                                                  ││  ║
║  │  │  ──────────────────────────────────────────────────────────────── ││  ║
║  │  │  STATUS:     NEW DETECTION      CONFIDENCE:  45.3% (low)          ││  ║
║  │  │  RSSI:       -85 dBm  ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   ││  ║
║  │  │  PROXIMITY:  FAR           (>5 meters estimated)                ││  ║
║  │  │  MOVEMENT:   STATIONARY                                          ││  ║
║  │  │  LAST SEEN:  5.1 seconds ago                                     ││  ║
║  │  │  NOTE:       Device not in database - possible unknown device    ││  ║
║  │  │                                                                     ││  ║
║  │  └─────────────────────────────────────────────────────────────────────┘│  ║
║  │                                                                         │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                                                                ║
║  ────────────────────────────────────────────────────────────────────────────  ║
║  Statistics: Packets processed: 1,247 | Detection rate: 98.2% | Runtime: 00:05  ║
║  ────────────────────────────────────────────────────────────────────────────  ║
║                                                                                ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

### 8.2 Signal Strength Visualization

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                        RSSI HISTORY PLOT                                       │
│                                                                                 │
│  dBm                                                                      Time  │
│   │                                                                           │
│  -40 ─┬───────────────────────────────────────────────────────────────────────   │
│       │                                                                     │   │
│       │    ████                                                             │   │
│       │   ██████  Device A                                                 │   │
│  -50 ─┼──███████─────────────                                             │   │
│       │ ███████████                                                         │   │
│       │████████████                                                         │   │
│       │                                                                        │   │
│  -60 ─┼─Device B ─▶                                                         │   │
│       │       ░░░░░░░░░░░░░░░░░░░                                         │   │
│       │          ░░░░░░░░░░░░░░░░░░░░░░                                   │   │
│  -70 ─┼─            ░░░░░░░░░░░░░░░░░░░░░░░░░░░                            │   │
│       │                   ░░░░░░░░░░░░░░░░░░░░░░░░░░░                       │   │
│       │                              ░░░░░░░░░░░░░░░░░░░░░░                  │   │
│  -80 ─┼─                                 ░░░░░░░░░░░░░░░░░░░░░░░░░░░        │   │
│       │                                              ░░░░░░░░░░░░░░░░░░░     │   │
│       │                                                         ░░░░░░░░░   │   │
│  -90 ─┼─                                                            ░░░░░─  │
│       │                                                                   │   │
│  -100 ┼───────────────────────────────────────────────────────────────────────   │
│       │                                                                         │
│       0s        10s        20s        30s        40s        50s        60s   │
│                                                                                 │
│  Legend:                                                                      │
│    ████ = Device A (NEAR, STATIONARY)                                        │
│    ░░░░ = Device B (FAR, MOVING AWAY)                                         │
│                                                                                 │
│  Note: Each point represents one detected packet.                             │
│        Dropouts indicate no signal detected during that period.                │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

### 8.3 Movement Indication Display

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                        MOVEMENT DETECTION                                      │
│                                                                                 │
│  Device: Device_A                                                               │
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                 │
│  CURRENT STATUS:                                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                                                                         │  │
│  │   RSSI NOW:     -62 dBm                                                │  │
│  │   RSSI 10s AGO: -58 dBm                                                │  │
│  │   CHANGE:      -4 dBm (decreasing)                                    │  │
│  │                                                                         │  │
│  │   MOVEMENT INDICATOR:                                                  │  │
│  │                                                                         │  │
│  │   AWAY ◀────────────┼─────────────▶ TOWARD                             │  │
│  │                      │                                                   │  │
│  │                      │                                                   │  │
│  │                   [HERE]                                                 │  │
│  │                  Slight                                               │  │
│  │                  decrease                                               │  │
│  │                  (likely stationary                                     │  │
│  │                   with noise)                                           │  │
│  │                                                                         │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  60-SECOND TREND:                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                                                                         │  │
│  │   -55 ┤                    ●●●●●                                       │  │
│  │       │               ●●●●●●●●●●                                        │  │
│  │   -60 ┤          ●●●●●●●●●●●●●●●●  ← Recent stabilization                │  │
│  │       │     ●●●●●●●●●●●●●●●●●●●●                                         │  │
│  │   -65 ┤●●●●●●●●●●●●●●●●●●●●●                                             │  │
│  │       │●●●●●●●●●●●●●●●●●                                                 │  │
│  │   -70 ┤●●●●●●                                                          │  │
│  │       │                                                                  │  │
│  │       └──────────────────────────────────────────────────────▶           │  │
│  │       0s      10s      20s      30s      40s      50s      60s         │  │
│  │                              Earlier ────────▶ Later                    │  │
│  │                                                                         │  │
│  │   CONCLUSION: Device was approaching, now appears stationary           │  │
│  │               (possibly arrived at destination ~3m away)                │  │
│  │                                                                         │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

### 8.4 Output Summary Table

| Output Type | Description | Example Value |
|------------|-------------|---------------|
| **Device ID** | Identified device name | "Device_A", "Remote_1" |
| **Confidence Score** | Classification certainty | 94.2% |
| **RSSI** | Signal strength | -62 dBm |
| **RSSI History** | Last N RSSI values | [-58, -59, -60, -61, -62] |
| **Proximity** | Distance estimate category | "NEAR" |
| **Proximity Detail** | Estimated distance range | "~1-2 meters" |
| **Movement Status** | Direction of movement | "MOVING_AWAY" |
| **Movement Detail** | Trend description | "Signal decreasing 3 dB over 30s" |
| **Last Seen** | Time since last detection | "0.3 seconds ago" |
| **Packet Count** | Detected packets count | 47 packets |

---

## 9. Applications

### 9.1 Security Applications

| Application | Description | Implementation |
|-------------|-------------|----------------|
| **Unauthorized Device Detection** | Detect unknown devices in secure zones | Flag any detection below confidence threshold |
| **Access Control Enhancement** | Verify identity of devices attempting to access secure areas | Cross-reference detected device with allowed list |
| **Secure Facility Monitoring** | Monitor for rogue transmitters | Continuous scanning with alerting |
| **Anti-Tailgating** | Ensure only authorized individuals access secure doors | Monitor devices moving with authorized personnel |

### 9.2 Anti-Theft Applications

| Application | Description | Implementation |
|-------------|-------------|----------------|
| **Equipment Tracking** | Track valuable equipment without active tags | Monitor known devices, alert if moved |
| **Asset Movement Detection** | Detect when assets are being relocated | Track RSSI changes indicating movement |
| **Vehicle Anti-Theft** | Monitor for theft attempts on vehicles | Track devices associated with vehicle |
| **Warehouse Security** | Monitor inventory movement | Track devices on valuable items |

### 9.3 IoT Authentication Applications

| Application | Description | Implementation |
|-------------|-------------|----------------|
| **Device Authentication** | Verify genuine IoT devices vs counterfeits | Match RF fingerprint to known device profile |
| **Network Access Control** | Only allow authenticated devices on network | Pre-authorize devices based on fingerprint |
| **Supply Chain Verification** | Verify product authenticity at checkpoints | Compare fingerprints with factory records |
| **Smart Home Security** | Ensure only registered devices connect | Fingerprint-based device verification |

### 9.4 Practical Use Case Scenarios

#### Scenario 1: Office Security
```
┌────────────────────────────────────────────────────────────────────────────────┐
│                         USE CASE: Office After-Hours Security                  │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  Setup:                                                                        │
│  - RTL-SDR positioned in office corridor                                       │
│  - 433 MHz remote controls monitored                                           │
│  - 5 known devices registered (employee key fobs)                              │
│                                                                                │
│  Operation:                                                                    │
│  1. System monitors continuously during off-hours                             │
│  2. When a device is detected:                                                │
│     - Identifies device using RF fingerprint                                  │
│     - Logs time and signal strength                                           │
│  3. If unknown device detected:                                               │
│     - Triggers alert (email/SMS)                                              │
│     - Records evidence (RSSI, timestamp)                                       │
│  4. If known device detected:                                                 │
│     - Verifies authorized presence                                            │
│     - Monitors for unusual behavior (staying too long, going to restricted     │
│       areas)                                                                  │
│                                                                                │
│  Benefit: Passive monitoring without requiring employees to carry additional    │
│           hardware or remember to activate anything                          │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

#### Scenario 2: Warehouse Asset Tracking
```
┌────────────────────────────────────────────────────────────────────────────────┐
│                         USE CASE: Warehouse Equipment Tracking               │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  Setup:                                                                        │
│  - RTL-SDR at central warehouse location                                       │
│  - Tag equipment with low-power 433 MHz transmitters (battery-powered)        │
│  - Register all equipment fingerprints                                          │
│                                                                                │
│  Operation:                                                                    │
│  1. System monitors all registered equipment                                   │
│  2. For each detected device:                                                  │
│     - Identifies specific item                                                 │
│     - Tracks RSSI over time                                                     │
│     - Detects if equipment is being moved (RSSI changes)                       │
│  3. If equipment moves significantly:                                          │
│     - Alert management                                                         │
│     - Track direction of movement                                              │
│  4. Daily reports:                                                             │
│     - Equipment locations (near/far from receiver)                             │
│     - Movement history                                                         │
│     - Any unauthorized relocations                                             │
│                                                                                │
│  Benefit: Track equipment without expensive active RFID infrastructure         │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

#### Scenario 3: Home Security
```
┌────────────────────────────────────────────────────────────────────────────────┐
│                         USE CASE: Home Security Enhancement                    │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  Setup:                                                                        │
│  - RTL-SDR connected to Raspberry Pi in home                                   │
│  - Register all household remote controls and IoT devices                      │
│  - Monitor 433 MHz garage door, alarm remotes, etc.                          │
│                                                                                │
│  Operation:                                                                    │
│  1. System knows authorized devices:                                           │
│     - Garage door remote (used daily)                                          │
│     - Car key fob (used when leaving)                                          │
│     - Neighbor's garage remote (should NOT be detected at high levels)         │
│  2. Detection events:                                                          │
│     - When garage door activated → check if it's authorized remote              │
│     - If unknown remote detected → possible break-in                           │
│  3. Tracking:                                                                  │
│     - Monitor RSSI to estimate distance                                       │
│     - If signal suddenly appears very strong → device very close to home      │
│                                                                                │
│  Benefit: Add RF fingerprinting layer to existing security systems             │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Challenges and Limitations

### 10.1 Technical Challenges

| Challenge | Description | Impact | Mitigation |
|-----------|-------------|--------|------------|
| **Signal Variability** | Same device produces slightly different signals each transmission | Reduced classification accuracy | Capture multiple samples; use statistical features |
| **Environmental Noise** | Other signals, interference, multipath | False detections | Bandpass filtering; SNR threshold |
| **Hardware Limitations** | RTL-SDR limited bandwidth, no calibrated RSSI | Limited precision | Use relative measurements; calibrate empirically |
| **Clock Drift** | RTL-SDR oscillator not perfectly stable | Affects CFO measurement | Temperature stabilization; calibration |
| **Multipath Fading** | Signals reflect off walls/objects | RSSI fluctuations | Signal averaging; ignore rapid fluctuations |

### 10.2 RF Fingerprinting Challenges

| Challenge | Description | Mitigation |
|-----------|-------------|------------|
| **Feature Extraction Accuracy** | Precisely measuring tiny hardware impairments | Use averaging over many samples; high sample rate |
| **Environmental Effects** | Temperature, humidity affect signal | Recalibrate periodically; use relative features |
| **Signal Modulation** | Different modulation types produce different features | Train on specific modulation types; separate models |
| **Transmit Power Variation** | Device battery level affects signal strength | Use normalized features; separate power from fingerprint |

### 10.3 RSSI Tracking Challenges

| Challenge | Description | Mitigation |
|-----------|-------------|------------|
| **Non-Line-of-Sight** | Walls/objects attenuate signal | Position antenna strategically; understand environment |
| **RSSI Fluctuations** | Indoor multipath causes rapid RSSI changes | Use moving average; larger detection windows |
| **No Distance Calibration** | RTL-SDR RSSI not calibrated | Create empirical RSSI-to-distance mapping |
| **Limited Precision** | RSSI gives rough distance estimate only | Use categories (near/far) not precise distance |

### 10.4 Practical Limitations

| Limitation | Description | Workaround |
|------------|-------------|-----------|
| **Single Point Tracking** | Only one SDR, cannot triangulate | RSSI only; direction not available |
| **Detection Range** | RTL-SDR range limited by antenna and environment | Use appropriate antenna; optimal placement |
| **Specific Frequency** | Must know target frequency in advance | Research target devices; scan multiple bands |
| **Training Required** | Need samples from each device | Collect training data during setup phase |
| **Processor Load** | Real-time processing is CPU intensive | Optimize code; reduce sample rate if needed |

### 10.5 Expected Performance

| Metric | Expected Value | Notes |
|--------|---------------|-------|
| **Identification Accuracy** | 85-95% | Under controlled conditions |
| **Proximity Detection** | NEAR/MEDIUM/FAR | Qualitative, not quantitative |
| **Movement Detection** | Correct direction ~80% | With appropriate window size |
| **Update Rate** | 1-5 Hz | Depends on signal activity |
| **Latency** | <1 second | From signal to output |

---

## 11. Future Scope

### 11.1 Immediate Improvements (Short-Term)

| Improvement | Description | Implementation Effort |
|-------------|-------------|---------------------|
| **Multiple SDR Nodes** | Use 2-3 RTL-SDRs for TDoA triangulation | Medium |
| **Deep Learning** | Replace sklearn with CNN for better features | High |
| **GUI Dashboard** | Create graphical interface for monitoring | Medium |
| **Multi-Band Support** | Support 433 MHz, 915 MHz, 2.4 GHz | Low |
| **GPS Integration** | Add location context for outdoor tracking | Low |

### 11.2 Advanced Tracking (Medium-Term)

| Feature | Description | Benefit |
|---------|-------------|---------|
| **TDoA Localization** | Time Difference of Arrival using multiple SDRs | 2D position estimate |
| **AoA Estimation** | Angle of Arrival using directional antenna | Direction tracking |
| **Kalman Filtering** | Smooth tracking with prediction | Better movement estimation |
| **Particle Filter** | Handle non-linear/non-Gaussian tracking | Improved accuracy |
| **SLAM Integration** | Simultaneous Localization and Mapping | Unknown environment tracking |

### 11.3 Multi-Device Tracking (Medium-Term)

| Feature | Description |
|---------|-------------|
| **Simultaneous Tracking** | Track multiple devices at once |
| **Track Association** | Maintain separate tracks for each device |
| **Collision Resolution** | Handle multiple devices transmitting simultaneously |
| **Group Detection** | Identify devices that move together |
| **Anomaly Detection** | Flag unusual movement patterns |

### 11.4 Smart City Integration (Long-Term)

| Application | Description |
|-------------|-------------|
| **Urban Asset Tracking** | Track municipal assets across city |
| **Public Safety** | Enhance emergency response with device tracking |
| **Transportation** | Monitor vehicle and cargo movement |
| **Environmental Monitoring** | Combine with sensor networks |
| **Infrastructure Security** | Monitor critical infrastructure access |

### 11.5 Research Extensions

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                         FUTURE RESEARCH DIRECTIONS                             │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  PHYSICAL LAYER ────────────────────────────────────────────────────────────  │
│       │                                                                          │
│       │  RF Fingerprinting                                                      │
│       ▼                                                                          │
│  CLASSIFICATION ─────────────────────────────────────────────────────────────  │
│       │                                                                          │
│       │  Current: Sklearn (RF, SVM, KNN)                                        │
│       ├──────────────────────────────────────────────────────────────────────┐  │
│       │                                                                      │  │
│       ▼                                                                      ▼  │
│  DEEP LEARNING ────────────────▶  ADVERSARIAL ML                               │
│       │                                    │                                   │
│       │  CNN for feature learning          │  Defend against fingerprint       │
│       │  RNN/LSTM for time-series          │  spoofing attacks                  │
│       │                                                                      │  │
│       ▼                                                                      │  │
│  TRACKING ─────────────────────────────────────────────────────────────────┘  │
│       │                                                                          │
│       │  Current: RSSI proximity                                               │
│       ├──────────────────────────────────────────────────────────────────────┐  │
│       │                                    │                                   │  │
│       ▼                                    ▼                                   │  │
│  TDoA / AoA ─────────────────────────▶  KALMAN / PARTICLE                      │  │
│       │                                                                      │  │
│       │  Multiple synchronized SDRs    │  Better state estimation              │  │
│       │  Hyperbola intersection        │  Handle noise and uncertainty        │  │
│       │                                                                      │  │
│       ▼                                                                      │  │
│  LOCALIZATION ─────────────────────────────────────────────────────────────┘  │
│       │                                                                          │
│       │  Current: Single point RSSI                                             │
│       ├──────────────────────────────────────────────────────────────────────┐  │
│       │                                    │                                   │  │
│       ▼                                    ▼                                   │  │
│  2D/3D POSITION ─────────────────────▶  SLAM                                   │
│       │                                                                      │  │
│       │  Multi-floor building           │  Unknown environment mapping         │  │
│       │  Centimeter accuracy (UWB)      │  Continuous tracking                 │  │
│       │                                                                      │  │
│       ▼                                                                      │  │
│  PRACTICAL DEPLOYMENT ─────────────────────────────────────────────────────┘  │
│       │                                                                          │
│       │  Edge Computing (Raspberry Pi, Jetson Nano)                             │
│       │  Cloud Integration for scalable tracking                               │
│       │  Commercial product development                                        │
│       │                                                                          │
│       ▼                                                                          │
│  SMART CITY INTEGRATION ────────────────────────────────────────────────────  │
│       │                                                                          │
│       │  City-wide sensor network                                              │
│       │  Urban mobility tracking                                               │
│       │  Public safety applications                                             │
│       │                                                                          │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 12. Innovation

### 12.1 Key Innovations

| Innovation | Description | Advantage |
|------------|-------------|-----------|
| **Hardware-Level Identity** | Uses physical hardware characteristics | Cannot be spoofed in software |
| **Passive Tracking** | No cooperation required from target device | Works on any transmitting device |
| **Combined Identification + Tracking** | Uniquely identifies AND monitors location | Knows WHO and WHERE |
| **Low-Cost Implementation** | Uses $40 hardware vs $1000+ commercial systems | Accessible to students/researchers |
| **RSSI-Based Movement Detection** | Tracks device movement without GPS/indoor positioning | Simple but effective |

### 12.2 Practical Implementation Advantages

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                         WHY THIS PROJECT IS INNOVATIVE                        │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │  1. PASSIVE & INVISIBLE                                                  │  │
│  │  ─────────────────────────────────────────────────────────────────────  │  │
│  │  • No modifications to tracked devices needed                           │  │
│  │  • Device doesn't know it's being tracked                               │  │
│  │  • Works on any device that transmits RF signals                        │  │
│  │                                                                         │  │
│  │  BEFORE:  Device must cooperate (Bluetooth ON, app installed, etc.)    │  │
│  │  NOW:     Device can be completely unaware                              │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │  2. HARDWARE-LEVEL SECURITY                                             │  │
│  │  ─────────────────────────────────────────────────────────────────────  │  │
│  │  • Based on physical imperfections in hardware                          │  │
│  │  • Cannot be changed through software                                   │  │
│  │  • Each device has unique "electronic fingerprint"                     │  │
│  │                                                                         │  │
│  │  MAC Address:    Can be changed in 1 line of code                       │  │
│  │  RF Fingerprint: Requires physical hardware modification                 │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │  3. COMBINED IDENTIFICATION + TRACKING                                  │  │
│  │  ─────────────────────────────────────────────────────────────────────  │  │
│  │  • Not just "is this device present?"                                   │  │
│  │  • But also "where is it and is it moving?"                              │  │
│  │  • Real-time RSSI monitoring for proximity and movement                │  │
│  │                                                                         │  │
│  │  OUTPUT: "Device A is here, about 2 meters away, and moving away"       │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │  4. LOW-COST ACCESSIBILITY                                               │  │
│  │  ─────────────────────────────────────────────────────────────────────  │  │
│  │  • Total hardware cost: $40-60                                         │  │
│  │  • vs Commercial systems: $1000-10000+                                 │  │
│  │  • Uses open-source software throughout                                 │  │
│  │  • Suitable for student projects and research                            │  │
│  │                                                                         │  │
│  │  ┌─────────────────┐      ┌─────────────────┐                          │  │
│  │  │   THIS PROJECT  │      │  COMMERCIAL     │                          │  │
│  │  │   ────────────  │      │  ────────────   │                          │  │
│  │  │   Hardware: $50 │      │  Hardware: $5K+  │                          │  │
│  │  │   Software: $0  │      │  Software: $10K │                          │  │
│  │  │   Total: $50    │      │  Total: $15K+   │                          │  │
│  │  └─────────────────┘      └─────────────────┘                          │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │  5. EDUCATIONAL VALUE                                                    │  │
│  │  ─────────────────────────────────────────────────────────────────────  │  │
│  │  • Covers multiple domains: RF engineering, signal processing, ML     │  │
│  │  • Hands-on with real hardware and software                             │  │
│  │  • Applicable to real-world problems                                    │  │
│  │  • Foundation for advanced research                                     │  │
│  │                                                                         │  │
│  │  DOMAINS COVERED:                                                        │  │
│  │  ├── Electronics: RTL-SDR hardware, antenna design                     │  │
│  │  ├── Communications: Signal processing, modulation, propagation        │  │
│  │  ├── Computer Science: Python, ML, real-time processing                │  │
│  │  └── Security: Physical layer authentication, fingerprinting            │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

### 12.3 Novel Combination

```
TRADITIONAL APPROACHES:
┌────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│   GPS Tracking           MAC Address           RFID Tags           Cameras    │
│        │                      │                    │                  │       │
│        ▼                      ▼                    ▼                  ▼       │
│   ┌─────────┐           ┌─────────┐          ┌─────────┐       ┌─────────┐  │
│   │ Location│           │Identity │          │Identity │       │Visual   │  │
│   │ ONLY    │           │ONLY     │          │ONLY     │       │ONLY     │  │
│   └─────────┘           └─────────┘          └─────────┘       └─────────┘  │
│                                                                                 │
│   Problem: Each method does ONE thing only                                      │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘

OUR APPROACH:
┌────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│              RF FINGERPRINTING + RSSI TRACKING                                  │
│                       │                 │                                      │
│                       ▼                 ▼                                      │
│              ┌──────────────┐   ┌──────────────┐                               │
│              │  IDENTITY    │   │  LOCATION    │                               │
│              │  (WHO?)      │   │  (WHERE?)    │                               │
│              │              │   │              │                               │
│              │ • CFO        │   │ • RSSI       │                               │
│              │ • I/Q Imbal  │   │ • Proximity  │                               │
│              │ • Phase Noise│   │ • Movement   │                               │
│              └──────┬───────┘   └──────┬───────┘                               │
│                     │                  │                                      │
│                     └────────┬─────────┘                                      │
│                              ▼                                                 │
│                    ┌──────────────────┐                                         │
│                    │   COMBINED      │                                         │
│                    │   OUTPUT        │                                         │
│                    │                 │                                         │
│                    │ WHO + WHERE +   │                                         │
│                    │ MOVEMENT        │                                         │
│                    └─────────────────┘                                         │
│                                                                                 │
│   Advantage: Comprehensive device knowledge in single system                  │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

### 12.4 Project Differentiation

| Aspect | This Project | Typical Student Project |
|--------|--------------|-------------------------|
| **Approach** | Physical layer fingerprinting | Software-only or simple threshold detection |
| **Hardware** | Real SDR hardware | Simulation only or expensive equipment |
| **ML Integration** | Real ML pipeline | Basic if any |
| **Tracking** | RSSI-based movement detection | Static presence only |
| **Output** | Real-time identification + tracking | Single classification result |
| **Cost** | $40-60 total | Often simulation-only |
| **Practicality** | Deployable proof-of-concept | Proof-of-concept only |

---

## Summary

This project implements an **Invisible Device Tracker** using **RF Fingerprinting** for identification and **RSSI monitoring** for basic tracking. The system demonstrates a practical, low-cost approach to device tracking that operates passively without requiring cooperation from tracked devices.

**Key Components:**
1. **RTL-SDR Hardware** ($25-35) - For RF signal capture
2. **Python Software** - Signal processing, feature extraction, ML classification
3. **RF Fingerprint Features** - CFO, I/Q imbalance, phase noise, etc.
4. **RSSI Tracking** - Proximity estimation and movement detection

**Expected Outcomes:**
- 85-95% device identification accuracy
- NEAR/MEDIUM/FAR proximity indication
- MOVING_TOWARD/MOVING_AWAY/STATIONARY status
- Real-time monitoring and alerting

**Innovation:**
- Combines identification and tracking in single low-cost system
- Uses hardware-level fingerprints impossible to spoof
- Passive operation requiring no device cooperation
- Accessible to students with ~$50 budget

**Learning Outcomes:**
- RF engineering fundamentals
- Digital signal processing
- Machine learning applications
- Real-time system implementation
- Hardware-software integration

---

*Document Version: 1.0*  
*Prepared for: Electronics and Communication Engineering Project*  
*Date: April 2026*

---

## Appendix A: Quick Start Checklist

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PROJECT QUICK START                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  □ Step 1: Hardware Setup                                                   │
│    ├─ Purchase RTL-SDR dongle ($25-35)                                      │
│    ├─ Get 433 MHz antenna ($10-15)                                          │
│    └─ Verify laptop/PC with USB 2.0                                        │
│                                                                              │
│  □ Step 2: Software Installation                                            │
│    ├─ Install Python 3.8+                                                  │
│    ├─ pip install numpy scipy scikit-learn matplotlib pyrtlsdr pandas      │
│    └─ Install RTL-SDR drivers                                               │
│                                                                              │
│  □ Step 3: Test Signal Reception                                            │
│    ├─ Connect RTL-SDR and antenna                                           │
│    ├─ Run basic reception test                                             │
│    └─ Verify I/Q sample capture                                             │
│                                                                              │
│  □ Step 4: Data Collection                                                  │
│    ├─ Prepare 2-3 devices to track                                         │
│    ├─ Capture training data for each device                                │
│    ├─ Capture negative samples (noise)                                     │
│    └─ Label and save dataset                                                │
│                                                                              │
│  □ Step 5: Feature Extraction                                                │
│    ├─ Implement feature extraction functions                                │
│    ├─ Extract features from training data                                   │
│    └─ Analyze feature distributions                                         │
│                                                                              │
│  □ Step 6: Model Training                                                   │
│    ├─ Split data into train/test sets                                      │
│    ├─ Train Random Forest classifier                                        │
│    ├─ Evaluate with cross-validation                                        │
│    └─ Save trained model                                                   │
│                                                                              │
│  □ Step 7: RSSI Tracker Implementation                                      │
│    ├─ Create tracker class with history buffer                             │
│    ├─ Implement proximity estimation                                        │
│    ├─ Implement movement detection                                          │
│    └─ Test with known devices                                               │
│                                                                              │
│  □ Step 8: Integration and Testing                                          │
│    ├─ Combine all modules into main application                            │
│    ├─ Test with known devices                                              │
│    ├─ Test with unknown devices                                             │
│    └─ Verify real-time operation                                            │
│                                                                              │
│  □ Step 9: Documentation                                                    │
│    ├─ Record experimental results                                          │
│    ├─ Create demonstration videos                                           │
│    └─ Prepare project report                                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Appendix B: Troubleshooting Guide

| Problem | Likely Cause | Solution |
|---------|-------------|----------|
| No signal detected | Wrong frequency | Verify target device frequency |
| Weak signals | Antenna not connected | Check SMA connector |
| High classification error | Insufficient training data | Collect more samples |
| RSSI very noisy | Indoor multipath | Use averaging, larger windows |
| Model predicts "Unknown" always | Threshold too high | Lower confidence threshold |
| SDR not detected | Driver not installed | Install Zadig drivers |
| Python crashes | Memory overflow | Reduce sample buffer size |

---

*End of Specification Document*
