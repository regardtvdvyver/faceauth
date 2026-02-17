---
name: Hardware Compatibility Report
about: Report your laptop/camera setup (working or not)
title: '[Hardware] '
labels: hardware-report
assignees: ''
---

## Laptop Model

(e.g., ThinkPad P14s Gen 5, Dell XPS 15 9530)

## IR Camera

- **Device path**: (e.g., `/dev/video2`)
- **v4l2-ctl output**:

```
Paste output of: v4l2-ctl --device=/dev/videoX --all
```

## RGB Camera

- **Device path**: (e.g., `/dev/video0`)

## GPU (if using OpenVINO)

- **Model**: (e.g., Intel Arc, Intel UHD, none)
- **OpenVINO provider**: (GPU / CPU / auto)

## Status

- [ ] Working
- [ ] Partial (describe what works and what doesn't below)
- [ ] Not Working

## Notes

Any additional observations -- performance, quirks, workarounds, etc.

## faceauth status output

```
Paste output of: faceauth status
```
