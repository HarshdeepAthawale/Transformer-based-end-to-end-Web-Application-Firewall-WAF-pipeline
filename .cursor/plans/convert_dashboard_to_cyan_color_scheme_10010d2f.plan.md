---
name: Convert Dashboard to Cyan Color Scheme
overview: Change all dashboard colors from multi-colored (blue, green, red, orange, amber) to a consistent cyan-based color scheme for Request Volume & Threats chart, metrics cards, alerts, and activity feed.
todos:
  - id: update-charts-section-colors
    content: Change Request Volume & Threats chart from blue/red to cyan colors in charts-section.tsx
    status: completed
  - id: update-metrics-cards-colors
    content: Remove colored borders from metrics cards (blue, green, red, amber) and apply cyan styling in metrics-overview.tsx
    status: completed
  - id: update-alerts-section-colors
    content: Remove red, amber, gray colors from alerts and apply cyan styling in alerts-section.tsx
    status: completed
  - id: update-activity-feed-colors
    content: Remove green and red colors from activity feed and apply cyan styling in activity-feed.tsx
    status: completed
isProject: false
---

# Cyan Color Scheme Implementation Plan

## Overview

Convert the WAF dashboard from a multi-colored scheme to a consistent cyan-based color palette, removing blue, green, red, orange, and amber colors as requested.

## Current Color Usage Analysis

- **Request Volume & Threats chart**: Blue (#2563eb) and red (#dc2626) strokes and fills
- **Metrics cards**: Blue, green, red, amber border colors for Total Requests, Threats Blocked, Attack Rate, Response Time
- **Alerts section**: Red, amber, gray border colors and severity indicators
- **Activity feed**: Green and red colors for blocked/allowed activities

## Implementation Steps

### 1. Update Charts Section Colors

Modify `frontend/components/charts-section.tsx`:

- Change Request Volume & Threats chart strokes from blue/red to cyan
- Update area fills and patterns to use cyan theme
- Ensure chart tooltips and legends use cyan colors

### 2. Update Metrics Cards Colors

Modify `frontend/components/metrics-overview.tsx`:

- Remove colored borders from all metric cards (blue, green, red, amber)
- Apply consistent cyan styling to all cards
- Maintain priority-based text colors but remove colored backgrounds

### 3. Update Alerts Section Colors

Modify `frontend/components/alerts-section.tsx`:

- Remove red, amber, and gray border colors from alerts
- Apply cyan border styling to all alert types
- Update severity indicators to use cyan theme
- Remove colored backgrounds from severity badges

### 4. Update Activity Feed Colors

Modify `frontend/components/activity-feed.tsx`:

- Remove green and red colors from activity items
- Apply cyan styling for both blocked and allowed activities
- Use cyan backgrounds and text colors consistently

## Color Mapping

- **Current multi-colors** → **New cyan scheme**
- Blue (#2563eb) → Cyan (#06b6d4 or similar)
- Green (#16a34a, #22c55e) → Cyan variants
- Red (#dc2626, #ef4444) → Cyan
- Orange (#ea580c) → Cyan
- Amber (#f59e0b) → Cyan

## Files to Modify

- `frontend/components/charts-section.tsx` - Chart colors and patterns
- `frontend/components/metrics-overview.tsx` - Metric card borders and styling
- `frontend/components/alerts-section.tsx` - Alert borders and severity colors
- `frontend/components/activity-feed.tsx` - Activity item colors

## Expected Outcome

Clean, consistent cyan-based color scheme across the entire dashboard while maintaining visual hierarchy and readability.