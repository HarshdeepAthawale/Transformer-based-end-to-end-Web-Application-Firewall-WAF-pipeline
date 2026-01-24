---
name: remove-dark-mode-light-only
overview: Remove all dark mode functionality from the WAF dashboard and keep only light mode, similar to Cloudflare's approach
todos:
  - id: remove-theme-components
    content: Delete theme-provider.tsx and theme-toggle.tsx components
    status: completed
  - id: update-layout
    content: Remove ThemeProvider wrapper and theme initialization script from layout.tsx
    status: completed
  - id: update-metadata
    content: Remove dark mode icon references from metadata in layout.tsx
    status: completed
  - id: clean-css
    content: Remove dark mode CSS variables and custom variant from globals.css
    status: completed
  - id: clean-ui-components
    content: "Remove all dark: prefixed classes from UI components"
    status: completed
  - id: remove-header-toggle
    content: Remove theme toggle from header component
    status: completed
  - id: remove-dependency
    content: Remove next-themes dependency from package.json
    status: completed
  - id: verify-cleanup
    content: Verify all dark mode references are removed and light mode works
    status: in_progress
isProject: false
---

## Overview

The current dashboard implements dark mode using next-themes with Tailwind CSS dark mode classes. To remove dark mode entirely and keep only light mode like Cloudflare, we need to:

1. **Remove theme-related components and dependencies**

- Delete `theme-provider.tsx` and `theme-toggle.tsx` components
- Remove next-themes dependency from package.json
- Remove theme toggle usage from header.tsx

2. **Update layout and configuration**

- Remove ThemeProvider wrapper from layout.tsx
- Remove theme initialization script from layout.tsx
- Update metadata to remove dark mode icon references

3. **Clean up CSS and UI components**

- Remove dark mode CSS variables from globals.css (lines 54-96)
- Remove all `dark:` prefixed classes from UI components
- Update CSS custom variant to remove dark mode support

4. **Verify and clean up**

- Ensure all dark mode references are removed
- Test that the application works in light mode only

## Files to modify:

- `frontend/app/layout.tsx` - Remove theme provider and initialization
- `frontend/app/globals.css` - Remove dark mode styles
- `frontend/components/header.tsx` - Remove theme toggle
- `frontend/package.json` - Remove next-themes dependency
- All UI components in `frontend/components/ui/` - Remove dark: classes

## Expected outcome:

The dashboard will only support light mode, with all dark mode functionality completely removed, matching Cloudflare's light-only approach.