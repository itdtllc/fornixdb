---
title: Home Lab Notes
description: Standing notes about the home network, devices, and routines.
---

A grab-bag of things I keep forgetting about the home setup. See [[Backups]]
before touching anything important.

# Network

The home network is on the 192.168.1.x range.

## Router

The router admin page is at 192.168.1.1. It is a Netgear unit in the hallway
closet; the firmware auto-updates on Sunday nights.

## Wi-Fi

There are two SSIDs: "Homestead" for everyday devices and "Homestead-IoT" for
the cameras and plugs, which are kept on a separate VLAN.

# Storage

## NAS

The Synology NAS lives at 192.168.1.50. It holds the family photos and the
Time Machine backups. Its web UI is on port 5000.

## Cameras

Three PoE cameras feed into the NAS Surveillance Station. They retain footage
for 14 days before it is overwritten.

# Backups

Time Machine runs to the NAS every night at 2:00 AM. A second offsite copy is
pushed to Backblaze every Saturday. Restore drills happen quarterly.

# Internet

The ISP is Sonic, on a 1 Gbps fiber line. Support is reached at 1-888-555-0100,
and the account number is on the paper bill in the kitchen drawer.
