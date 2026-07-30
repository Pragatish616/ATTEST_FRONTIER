"""Materializes the ~30-document Northwind Devices demo corpus onto disk.

Run once (idempotent — safe to re-run, it just overwrites) to (re)populate
`demo/corpus/*.md` before running `demo/run_demo.py`. Keeping the corpus as
a generated artifact rather than hand-maintained files makes the two
deliberately-seeded documents (see `demo/SEED_NOTES.md`) easy to audit in
one place: the DOCS dict below.

Domain: a fictional consumer-tech company, "Northwind Devices" (earbuds,
smartwatches, a home hub, and a cross-platform support app), covering
product specs, company policy, support KB articles, and release notes —
one coherent corpus a support/RAG bot would plausibly be built over.

Two seeds (PLAN.md §8 hour 14-18 gate — see SEED_NOTES.md for the full
rationale):
  (a) doc 19 (`19-kb-supportbot-desktop-requirements.md`) asserts Windows 10
      "remains a fully supported, secure choice" — true when written,
      factually stale now that Microsoft ended standard support for
      Windows 10 on 2025-10-14 (a real, web-verifiable, fixed historical
      fact) -> targets verdict STALE.
  (b) doc 01 (`01-spec-aria-earbuds-gen2.md`) documents fast USB-C wired
      charging for the earbuds case but never mentions wireless/Qi
      charging anywhere in the corpus -> targets verdict UNSUPPORTED (and
      is a good FRAGILE probe target, see SEED_NOTES.md).
"""

from __future__ import annotations

from pathlib import Path

CORPUS_DIR = Path(__file__).parent / "corpus"

DOCS: dict[str, str] = {
    # -------------------------------------------------------------------
    # Product spec sheets
    # -------------------------------------------------------------------
    "01-spec-aria-earbuds-gen2.md": """\
# Northwind Aria Earbuds (2nd Gen) — Spec Sheet

**Category:** True wireless earbuds
**Doc ID:** doc-01

## Overview
The Aria Earbuds (2nd Gen) are Northwind's mid-range true wireless earbuds,
built for everyday commuting and workouts.

## Audio
- Dynamic 10mm drivers, tuned for balanced bass response.
- Active noise cancellation (ANC) with a dedicated transparency mode.
- Supports SBC and AAC codecs.

## Battery & Charging
- Earbuds: up to 6 hours of playback per charge with ANC on.
- Charging case: adds 3 additional full charges (24 hours total).
- The case charges via USB-C. A 10-minute fast charge via the USB-C port
  provides approximately 1 hour of additional playback.
- Battery level is shown via an LED indicator on the front of the case.

## Connectivity
- Bluetooth 5.3, with multipoint pairing to two devices at once.
- IPX4 sweat and splash resistance on the earbuds (the case is not
  water resistant).

## In the Box
- Aria Earbuds (2nd Gen), charging case, USB-C to USB-C cable, three sizes
  of silicone ear tips.
""",
    "02-spec-aria-earbuds-pro.md": """\
# Northwind Aria Earbuds Pro — Spec Sheet

**Category:** True wireless earbuds (premium)
**Doc ID:** doc-02

## Overview
The Aria Earbuds Pro are Northwind's flagship earbuds, adding adaptive ANC
and spatial audio on top of the Aria (2nd Gen) feature set.

## Audio
- Custom 11mm drivers with adaptive ANC that adjusts to ambient noise
  every 500ms.
- Spatial audio with head tracking.
- Supports SBC, AAC, and LDAC codecs.

## Battery & Charging
- Earbuds: up to 7 hours of playback per charge with ANC on.
- Charging case: adds 4 additional full charges (28 hours total).
- The case charges via USB-C only. A 5-minute wired fast charge provides
  approximately 1 hour of playback.

## Connectivity
- Bluetooth 5.3, multipoint pairing to two devices.
- IPX4 sweat and splash resistance on the earbuds.

## In the Box
- Aria Earbuds Pro, charging case, USB-C to USB-C cable, four sizes of
  silicone ear tips.
""",
    "03-spec-comet-watch-se.md": """\
# Northwind Comet Watch SE — Spec Sheet

**Category:** Smartwatch (entry-level)
**Doc ID:** doc-03

## Overview
Comet Watch SE is Northwind's entry-level smartwatch, running CometOS.

## Display
- 1.5" AMOLED, always-on display option (reduces battery life by ~20%).
- Corning-hardened glass, scratch resistant.

## Health & Fitness
- Heart rate sensor, SpO2 sensor, sleep tracking.
- 24 built-in workout modes.
- Not depth-rated for swimming; splash resistant only (IPX4).

## Battery
- Up to 3 days typical use, 1.5 days with always-on display enabled.
- Charges fully in 90 minutes via the included magnetic puck charger.

## Connectivity
- Bluetooth 5.1, companion app for iOS and Android.
- No built-in GPS; uses connected-phone GPS for outdoor tracking.

## Compatibility
- Requires the Northwind Comet app, iOS 15+ or Android 10+.
""",
    "04-spec-comet-watch-pro.md": """\
# Northwind Comet Watch Pro — Spec Sheet

**Category:** Smartwatch (premium)
**Doc ID:** doc-04

## Overview
Comet Watch Pro adds built-in GPS, swim tracking, and a titanium case to
the Comet lineup, running CometOS.

## Display
- 1.7" AMOLED, always-on display standard.
- Sapphire crystal cover glass.

## Health & Fitness
- Heart rate, SpO2, skin temperature, and sleep tracking.
- 40 built-in workout modes including open-water swim tracking.
- Water resistant to 50 meters (swim-proof).

## Battery
- Up to 5 days typical use, 2 days with GPS tracking used daily.
- Charges fully in 75 minutes via the included magnetic puck charger.

## Connectivity
- Bluetooth 5.2, built-in dual-band GPS/GNSS.
- Companion app for iOS and Android.

## Compatibility
- Requires the Northwind Comet app, iOS 15+ or Android 10+.
""",
    "05-spec-zephyr-hub.md": """\
# Northwind Zephyr Hub — Spec Sheet

**Category:** Smart home hub
**Doc ID:** doc-05

## Overview
Zephyr Hub is a home-network hub that bridges Northwind devices (Aria,
Comet, Pulse, Halo) over Thread and Bluetooth to Wi-Fi.

## Connectivity
- Wi-Fi 6, Thread border router, Bluetooth 5.2.
- Supports up to 64 paired Northwind devices simultaneously.

## Hardware
- 2x USB-C ports (one for power, one for accessory charging passthrough).
- Built-in speaker for voice prompts and Find My Hub chirps.

## Setup
- Configured via the Northwind Home app (iOS 15+, Android 10+).
- Requires a 2.4GHz or 5GHz home Wi-Fi network during setup.

## Power
- Powered via included USB-C wall adapter; no battery, always plugged in.
""",
    "06-spec-zephyr-dock-mini.md": """\
# Northwind Zephyr Dock Mini — Spec Sheet

**Category:** Charging dock
**Doc ID:** doc-06

## Overview
Zephyr Dock Mini is a compact 2-in-1 charging dock for one Comet Watch and
one pair of Aria Earbuds.

## Charging
- One Qi wireless charging puck for Comet Watch (up to 5W).
- One USB-C port for charging an Aria Earbuds case (any generation).

## Hardware
- USB-C input, ships with a 20W USB-C wall adapter.
- Non-slip weighted base.

## Compatibility
- Compatible with all Comet Watch models.
- Compatible with all Aria Earbuds case generations via the USB-C port.
""",
    "07-spec-nimbus-speaker.md": """\
# Northwind Nimbus Speaker — Spec Sheet

**Category:** Portable Bluetooth speaker
**Doc ID:** doc-07

## Overview
Nimbus is a portable, water-resistant Bluetooth speaker with a 360-degree
sound profile.

## Audio
- Dual 20mm tweeters, single 50mm woofer.
- 360-degree sound dispersion.

## Battery
- Up to 14 hours playback at moderate volume.
- Charges via USB-C, full charge in 2.5 hours.

## Durability
- IP67 rated: dust-tight and withstands submersion to 1 meter for 30
  minutes.

## Connectivity
- Bluetooth 5.3, pairs with two Nimbus speakers for stereo pairing.
""",
    "08-spec-pulse-fitness-band.md": """\
# Northwind Pulse Fitness Band — Spec Sheet

**Category:** Fitness band
**Doc ID:** doc-08

## Overview
Pulse is a lightweight fitness band focused on activity and sleep tracking,
without a full smartwatch display.

## Display
- 0.9" monochrome OLED, tap-to-wake.

## Health & Fitness
- Heart rate and step tracking, sleep stage tracking.
- Water resistant to 30 meters.

## Battery
- Up to 9 days typical use.
- Charges via a proprietary 2-pin magnetic clip, full charge in 100
  minutes.

## Connectivity
- Bluetooth Low Energy, companion app for iOS and Android.
""",
    "09-spec-halo-smart-display.md": """\
# Northwind Halo Smart Display — Spec Sheet

**Category:** Smart display / hub accessory
**Doc ID:** doc-09

## Overview
Halo is a countertop smart display that shows notifications, calendar, and
health summaries from paired Northwind devices via Zephyr Hub.

## Display
- 7" IPS touchscreen, 800x480 resolution.
- Ambient light sensor for auto-brightness.

## Connectivity
- Requires a Zephyr Hub on the same network; does not connect to Northwind
  devices directly.
- Wi-Fi 5, Bluetooth 5.0 (for initial setup only).

## Power
- Powered via included USB-C wall adapter; no internal battery.
""",
    "10-spec-drift-travel-charger.md": """\
# Northwind Drift Travel Charger — Spec Sheet

**Category:** Multi-device travel charger
**Doc ID:** doc-10

## Overview
Drift is a foldable travel charger with three outputs for charging Comet
Watch, Aria Earbuds, and a phone simultaneously.

## Charging
- One Qi puck for Comet Watch (5W).
- One USB-C port for an Aria Earbuds case (any generation).
- One USB-C PD port for phones (up to 20W).

## Hardware
- Folds flat for travel; magnetic alignment on the Qi surface.
- Ships with a compact 30W GaN wall adapter.
""",
    # -------------------------------------------------------------------
    # Company policy
    # -------------------------------------------------------------------
    "11-policy-returns-refunds.md": """\
# Returns & Refunds Policy

**Doc ID:** doc-11

Northwind Devices accepts returns of unopened or opened products within
**30 days** of the delivery date for a full refund to the original payment
method.

- Items must be returned with all original accessories and packaging where
  reasonably possible.
- Refunds are issued within 5-7 business days of Northwind receiving the
  returned item.
- Return shipping is free for defective items; a flat $6.99 return
  shipping fee applies to non-defective returns.
- Final sale / clearance items are not eligible for return.

This is the standard policy for all individual consumer purchases made
directly through northwinddevices.com or an authorized retailer.
""",
    "12-policy-warranty-terms.md": """\
# Warranty Terms

**Doc ID:** doc-12

All Northwind Devices hardware (Aria, Comet, Zephyr, Nimbus, Pulse, Halo,
Drift) carries a **1-year limited warranty** from the original date of
purchase, covering manufacturing defects in materials and workmanship.

- The warranty does not cover accidental damage, water damage beyond the
  product's rated resistance, or normal battery capacity degradation.
- Warranty service is either repair or replacement, at Northwind's
  discretion.
- Proof of purchase is required for all warranty claims.
- The warranty is non-transferable and applies only to the original
  purchaser.
""",
    "13-policy-shipping-delivery.md": """\
# Shipping & Delivery Policy

**Doc ID:** doc-13

- Standard shipping (5-7 business days) is free on orders over $35.
- Expedited shipping (2-3 business days) is available for a flat $12.99.
- Orders placed before 1pm ET on a business day ship the same day.
- Northwind currently ships to the United States, Canada, and the United
  Kingdom.
- Tracking information is emailed automatically once a package ships.
""",
    "14-policy-trade-in-program.md": """\
# Trade-In Program

**Doc ID:** doc-14

Customers may trade in an eligible used device (any condition) toward the
purchase of a new Northwind product.

- Trade-in credit is issued as a Northwind gift card, applied at checkout.
- Eligible devices: any generation of Aria Earbuds, Comet Watch, or Pulse
  Fitness Band, from Northwind or another manufacturer.
- Trade-in value is determined at checkout based on device model and
  condition (Good / Fair / Poor).
- Devices must be shipped to Northwind within 14 days of the trade-in
  quote being generated, using the provided prepaid label.
""",
    "15-policy-extended-care-plan.md": """\
# Extended Care Plan

**Doc ID:** doc-15

The Extended Care Plan is an optional paid add-on purchased at checkout
that extends coverage beyond the standard 1-year limited warranty.

- Extends coverage to 2 years total from date of purchase.
- Adds coverage for up to two incidents of accidental damage per year,
  subject to a $25 service fee per incident.
- Must be purchased within 30 days of the original device purchase.
- Not available for Halo Smart Display or Zephyr Hub (fixed home
  installations are excluded from this plan).
""",
    "16-policy-business-bulk-orders.md": """\
# Business & Bulk Order Policy

**Doc ID:** doc-16

Northwind offers a Business Program for orders of 10 or more units of the
same product.

- Volume discount tiers: 5% off for 10-24 units, 10% off for 25-99 units,
  custom pricing for 100+ units (contact business sales).
- Bulk orders are invoiced net-30 for approved business accounts.
- Bulk orders ship via freight for quantities over 50 units and may take
  10-14 business days.
- Business Program orders are not eligible for the standard 30-day
  consumer return window; instead they follow the terms in the signed
  business order agreement.
""",
    "17-policy-data-privacy.md": """\
# Data & Privacy Policy (Summary)

**Doc ID:** doc-17

- Health and fitness data collected by Comet Watch and Pulse Fitness Band
  is stored encrypted at rest and is not sold to third parties.
- Users may export or permanently delete their account data at any time
  from the companion app's Privacy settings.
- Northwind retains anonymized, aggregated usage analytics (e.g. feature
  usage counts) even after account deletion, with no personally
  identifying information attached.
- Location data from Comet Watch Pro's built-in GPS is stored on-device
  and only uploaded if the user explicitly enables cloud workout sync.
""",
    "18-policy-accessibility-statement.md": """\
# Accessibility Statement

**Doc ID:** doc-18

Northwind Devices is committed to making its products and companion apps
usable by people with disabilities.

- The Northwind Home, Comet, and SupportBot apps support VoiceOver (iOS)
  and TalkBack (Android) screen readers.
- Comet Watch supports adjustable text size and a high-contrast display
  mode.
- Customers needing accommodations for the standard return process
  (e.g. extended return windows for accessibility-related exchanges) can
  contact accessibility-support@northwinddevices.com.
""",
    # -------------------------------------------------------------------
    # Support KB (doc 19 is the deliberate STALE seed)
    # -------------------------------------------------------------------
    "19-kb-supportbot-desktop-requirements.md": """\
# SupportBot Desktop — System Requirements

**Doc ID:** doc-19

SupportBot Desktop is Northwind's desktop help-center application, used to
run diagnostics on paired Aria, Comet, and Zephyr devices from a PC.

## Minimum Requirements
- **Operating system:** Windows 10 (64-bit) or later, or macOS 12 or
  later.
- 4GB RAM minimum, 8GB recommended.
- 500MB free disk space.
- USB-C or Bluetooth 5.0+ for device pairing.

## Windows Support Notes
Windows 10 continues to receive regular security patches directly from
Microsoft, so it remains a fully supported, secure choice for running
SupportBot Desktop. Northwind recommends keeping Windows Update enabled to
receive these patches automatically. No action is needed for Windows 10
users beyond staying current on Windows Update.

## macOS Support Notes
Northwind supports the three most recent annual macOS releases at any
given time.
""",
    "20-kb-supportbot-mobile-requirements.md": """\
# SupportBot Mobile — System Requirements

**Doc ID:** doc-20

SupportBot Mobile mirrors SupportBot Desktop's diagnostics for iOS and
Android.

## Minimum Requirements
- iOS 16 or later, or Android 11 or later.
- Bluetooth 5.0+ required for device pairing.
- Approximately 150MB of storage for the app and diagnostic logs.

## Notes
- Push notifications must be enabled to receive firmware-update alerts
  for paired devices.
- Background Bluetooth scanning must be permitted for automatic device
  detection.
""",
    "21-kb-comet-firmware-update-guide.md": """\
# Updating Comet Watch Firmware

**Doc ID:** doc-21

1. Open the Northwind Comet app and ensure your watch is paired and within
   Bluetooth range.
2. Go to **Device > Firmware Update**. If an update is available, tap
   **Download**.
3. Keep the watch above 30% battery and within range during the update;
   installation takes 5-10 minutes and the watch will restart once.
4. Do not disconnect Bluetooth or close the app during installation — an
   interrupted update may require a factory reset to recover.

Firmware updates are released independently of CometOS major version
updates; see the relevant release notes doc for what changed in a specific
version.
""",
    "22-kb-bluetooth-pairing-troubleshooting.md": """\
# Bluetooth Pairing Troubleshooting

**Doc ID:** doc-22

If a Northwind device won't pair:

1. Confirm the device is charged above 20% — earbuds and watches will not
   enter pairing mode below this threshold.
2. Forget the device from your phone's Bluetooth settings, then re-add it
   from within the relevant Northwind app rather than the phone's system
   Bluetooth menu.
3. Reset the device: for Aria Earbuds, hold the case button for 10 seconds
   until the LED flashes white; for Comet Watch, hold the side button for
   15 seconds.
4. Ensure no more than 2 devices are already multipoint-connected to Aria
   Earbuds Pro/2nd Gen — a third connection attempt will fail silently.
""",
    "23-kb-aria-firmware-changelog.md": """\
# Aria Earbuds Firmware Changelog (Summary)

**Doc ID:** doc-23

- **v2.5** — Improved ANC adjustment speed on Aria Pro; fixed a bug where
  transparency mode occasionally didn't disable during phone calls.
- **v2.4** — Added multipoint pairing support to Aria (2nd Gen), matching
  Aria Pro.
- **v2.3** — Battery reporting accuracy improvements in the case LED
  indicator.
- **v2.2** — Initial public firmware release for Aria (2nd Gen).

Firmware updates are delivered automatically via the Northwind Home or
Comet app when the earbuds are in their case and connected.
""",
    "24-kb-zephyr-hub-setup-guide.md": """\
# Setting Up Zephyr Hub

**Doc ID:** doc-24

1. Plug in Zephyr Hub using the included USB-C adapter; the status light
   will pulse blue when ready to pair.
2. Open the Northwind Home app and select **Add Device > Zephyr Hub**.
3. Connect the hub to your 2.4GHz or 5GHz home Wi-Fi network when
   prompted.
4. Once online, the hub will automatically discover any Northwind devices
   already paired to your account within Bluetooth/Thread range.
5. A solid blue status light indicates the hub is online and connected to
   Northwind's cloud service.
""",
    "25-kb-account-login-faq.md": """\
# Account & Login FAQ

**Doc ID:** doc-25

**Q: I forgot my password. How do I reset it?**
Use "Forgot Password" on the sign-in screen of any Northwind app; a reset
link is emailed and expires after 1 hour.

**Q: Can I use one Northwind account across multiple devices?**
Yes — a single account can have unlimited Northwind devices paired to it
across the Comet, Home, and SupportBot apps.

**Q: How do I transfer a device to a new owner?**
Remove the device from your account under **Settings > My Devices >
Remove**, which also clears any Activation Lock so the new owner can pair
it fresh.
""",
    "26-kb-battery-health-faq.md": """\
# Battery Health FAQ

**Doc ID:** doc-26

**Q: Will fast charging degrade my earbuds' battery faster?**
Northwind's fast-charging circuitry (wired on Aria 2nd Gen, wired or
wireless on Aria Pro) is temperature-regulated and rated for the device's
full 1-year warranty period under normal use.

**Q: My Comet Watch battery drains faster after a firmware update — is
that normal?**
A one-time re-calibration of the battery percentage indicator after a
firmware update is normal and typically resolves within 2 full charge
cycles.

**Q: Does leaving earbuds in the case at 100% harm the battery?**
No — Northwind's charging cases stop drawing power once the earbuds reach
100% and only trickle-charge to maintain the level.
""",
    # -------------------------------------------------------------------
    # Release notes
    # -------------------------------------------------------------------
    "27-release-notes-supportbot-desktop-v4.2.md": """\
# SupportBot Desktop v4.2 — Release Notes

**Doc ID:** doc-27

- Added a dark mode option under Settings > Appearance.
- Improved diagnostic scan time for Zephyr Hub by roughly 40%.
- Fixed a crash that occurred when running a diagnostic scan while a Comet
  Watch firmware update was in progress.
- Minimum supported operating systems are unchanged from prior releases;
  see the SupportBot Desktop system requirements doc for current details.
""",
    "28-release-notes-supportbot-desktop-v4.1.md": """\
# SupportBot Desktop v4.1 — Release Notes

**Doc ID:** doc-28

- Added support for diagnosing Nimbus Speaker connection issues.
- Fixed an issue where exported diagnostic logs were missing timestamps.
- General performance improvements to app startup time.
""",
    "29-release-notes-cometos-3.0.md": """\
# CometOS 3.0 — Release Notes

**Doc ID:** doc-29

- New always-on watch face gallery with 12 additional faces.
- Sleep tracking now detects and logs naps under 90 minutes.
- Battery usage breakdown added under Settings > Battery.
- Requires Comet Watch SE or Comet Watch Pro; not available for earlier
  Comet Watch generations.
""",
    "30-release-notes-aria-firmware-2.5.md": """\
# Aria Earbuds Firmware v2.5 — Full Release Notes

**Doc ID:** doc-30

## Aria Earbuds Pro
- Improved adaptive ANC response time from 800ms to 500ms.
- Fixed transparency mode not disabling automatically during phone calls
  in some cases.

## Aria Earbuds (2nd Gen)
- Minor Bluetooth stability improvements for multipoint pairing.

Both models: this update is delivered automatically the next time the
earbuds are placed in their case with the case connected to power or a
paired phone nearby.
""",
}


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, content in DOCS.items():
        (CORPUS_DIR / filename).write_text(content, encoding="utf-8")
    print(f"Wrote {len(DOCS)} documents to {CORPUS_DIR}")


if __name__ == "__main__":
    main()
