'use client'

import { useEffect, useRef, useState } from 'react'
import Script from 'next/script'
import { useTheme } from 'next-themes'
import { cn } from '@/lib/utils'
import { getCountryCoordinates } from '@/lib/country-coordinates'
import type { GeoStats } from '@/lib/api'

const LEAFLET_CSS = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css'
const LEAFLET_JS = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'

const DARK_TILE_URL = 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
const DARK_TILE_ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/">CARTO</a>'

const STADIA_BASE =
  'https://tiles.stadiamaps.com/tiles/alidade_smooth/{z}/{x}/{y}{r}.png'
const getLightTileUrl = () => {
  const key = typeof process !== 'undefined' && process.env.NEXT_PUBLIC_STADIA_MAPS_API_KEY
  return key ? `${STADIA_BASE}?api_key=${key}` : STADIA_BASE
}
const LIGHT_TILE_ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://stadiamaps.com/">Stadia Maps</a>'

const MIN_RADIUS = 3
const MAX_RADIUS = 14
const DEFAULT_CENTER: [number, number] = [20, 0]
const DEFAULT_ZOOM = 2
const MIN_ZOOM = 2
const MAX_ZOOM = 19
const WORLD_BOUNDS: [[number, number], [number, number]] = [[-85, -180], [85, 180]]

const DARK_MAP_OVERRIDES = `
  .geo-attack-map-dark .leaflet-control-zoom {
    border: none !important;
  }
  .geo-attack-map-dark .leaflet-control-zoom a {
    background: #1e293b !important;
    color: #94a3b8 !important;
    border: 1px solid #334155 !important;
    border-bottom: none !important;
  }
  .geo-attack-map-dark .leaflet-control-zoom a:last-child {
    border-bottom: 1px solid #334155 !important;
  }
  .geo-attack-map-dark .leaflet-control-zoom a:hover {
    background: #334155 !important;
    color: #e2e8f0 !important;
  }
  .geo-attack-map-dark .leaflet-tooltip {
    background: #1e293b !important;
    color: #e2e8f0 !important;
    border: 1px solid #334155 !important;
  }
  .geo-attack-map-dark .leaflet-tooltip::before {
    border-top-color: #1e293b !important;
  }
`

declare global {
  interface Window {
    L: typeof import('leaflet')
  }
}

export function GeoAttackMap({
  stats,
  className,
}: {
  stats: GeoStats[]
  className?: string
}) {
  const { resolvedTheme } = useTheme()
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<ReturnType<Window['L']['map']> | null>(null)
  const tileLayerRef = useRef<ReturnType<Window['L']['tileLayer']> | null>(null)
  const markersRef = useRef<ReturnType<Window['L']['circleMarker']>[]>([])
  const [scriptReady, setScriptReady] = useState(false)

  const isDark = (resolvedTheme ?? 'dark') === 'dark'

  useEffect(() => {
    const link = document.createElement('link')
    link.rel = 'stylesheet'
    link.href = LEAFLET_CSS
    document.head.appendChild(link)
    return () => {
      document.head.removeChild(link)
    }
  }, [])

  useEffect(() => {
    if (!scriptReady || !containerRef.current || typeof window === 'undefined' || !window.L) return
    const L = window.L

    const destroyMap = () => {
      markersRef.current.forEach((m) => m.remove())
      markersRef.current = []
      tileLayerRef.current?.remove()
      tileLayerRef.current = null
      if (mapRef.current) {
        mapRef.current.remove()
        mapRef.current = null
      }
    }

    destroyMap()

    const map = L.map(containerRef.current, {
      preferCanvas: true,
      minZoom: MIN_ZOOM,
      maxZoom: MAX_ZOOM,
      maxBounds: L.latLngBounds(WORLD_BOUNDS[0], WORLD_BOUNDS[1]),
      maxBoundsViscosity: 1,
    }).setView(DEFAULT_CENTER, DEFAULT_ZOOM)

    const isDarkTile = (resolvedTheme ?? 'dark') === 'dark'
    const url = isDarkTile ? DARK_TILE_URL : getLightTileUrl()
    const attribution = isDarkTile ? DARK_TILE_ATTRIBUTION : LIGHT_TILE_ATTRIBUTION
    const opts = isDarkTile
      ? { attribution, subdomains: 'abcd' as const, maxZoom: MAX_ZOOM }
      : { attribution, maxZoom: MAX_ZOOM }
    tileLayerRef.current = L.tileLayer(url, opts).addTo(map)
    mapRef.current = map

    return () => {
      destroyMap()
    }
  }, [scriptReady, resolvedTheme])

  useEffect(() => {
    const L = window.L
    const map = mapRef.current
    if (!L || !map || !Array.isArray(stats)) return

    markersRef.current.forEach((m) => m.remove())
    markersRef.current = []

    const withBlocked = stats.filter((s) => s.blocked_requests > 0)
    const maxBlocked = Math.max(1, ...withBlocked.map((s) => s.blocked_requests))

    withBlocked.forEach((stat) => {
      const coords = getCountryCoordinates(stat.country_code)
      if (!coords) return
      const ratio = stat.blocked_requests / maxBlocked
      const radius = MIN_RADIUS + ratio * (MAX_RADIUS - MIN_RADIUS)
      const circle = L.circleMarker(coords, {
        radius,
        fillColor: '#fb5015',
        color: '#ea4c1f',
        weight: 1,
        opacity: 1,
        fillOpacity: 0.7,
      })
      circle.bindTooltip(
        `${stat.country_name} (${stat.country_code}): ${stat.blocked_requests.toLocaleString()} blocked`,
        { permanent: false, direction: 'top' }
      )
      circle.addTo(map)
      markersRef.current.push(circle)
    })
  }, [scriptReady, resolvedTheme, stats])

  return (
    <>
      <Script
        src={LEAFLET_JS}
        strategy="afterInteractive"
        onLoad={() => setScriptReady(true)}
      />
      <style dangerouslySetInnerHTML={{ __html: DARK_MAP_OVERRIDES }} />
      <div
        ref={containerRef}
        className={cn(className, isDark && 'geo-attack-map-dark')}
        style={{ height: '100%', width: '100%', minHeight: 300 }}
      />
    </>
  )
}
