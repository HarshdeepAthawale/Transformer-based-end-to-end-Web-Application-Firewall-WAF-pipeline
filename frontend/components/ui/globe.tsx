"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import createGlobe, { type COBEOptions } from "cobe"
import { useMotionValue, useSpring } from "motion/react"

import { cn } from "@/lib/utils"

const MOVEMENT_DAMPING = 800
const SCALE_MIN = 0.5
const SCALE_MAX = 1.8
const DEFAULT_SCALE = 0.95
const DEFAULT_THETA = 0.25

export interface GlobeMarkerWithLabel {
  location: [number, number]
  size: number
  label?: string
}

const GLOBE_CONFIG: COBEOptions = {
  width: 800,
  height: 800,
  onRender: () => {},
  devicePixelRatio: 2,
  phi: 0,
  theta: DEFAULT_THETA,
  dark: 0,
  diffuse: 0.4,
  mapSamples: 16000,
  mapBrightness: 1.2,
  scale: DEFAULT_SCALE,
  baseColor: [1, 1, 1],
  markerColor: [251 / 255, 100 / 255, 21 / 255],
  glowColor: [1, 1, 1],
  markers: [],
}

export function Globe({
  className,
  config: configProp,
}: {
  className?: string
  config?: Partial<COBEOptions>
}) {
  const config = { ...GLOBE_CONFIG, ...configProp }
  const phiRef = useRef(0)
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 })
  const [hoveredMarker, setHoveredMarker] = useState<string | null>(null)
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 })
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const pointerInteracting = useRef<{ x: number; y: number } | null>(null)

  const rPhi = useMotionValue(0)
  const rTheta = useMotionValue(0)
  const rsPhi = useSpring(rPhi, { mass: 0.5, damping: 25, stiffness: 120 })
  const rsTheta = useSpring(rTheta, { mass: 0.5, damping: 25, stiffness: 120 })
  const scaleMotion = useMotionValue(DEFAULT_SCALE)
  const scaleSpring = useSpring(scaleMotion, { mass: 0.3, damping: 20, stiffness: 150 })

  const updatePointerInteraction = useCallback((value: { x: number; y: number } | null) => {
    pointerInteracting.current = value
    if (canvasRef.current) {
      canvasRef.current.style.cursor = value !== null ? "grabbing" : "grab"
    }
  }, [])

  const handlePointerDown = useCallback(
    (e: React.PointerEvent) => {
      pointerInteracting.current = { x: e.clientX, y: e.clientY }
      updatePointerInteraction(pointerInteracting.current)
    },
    [updatePointerInteraction]
  )

  const handlePointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (pointerInteracting.current !== null) {
        const deltaX = e.clientX - pointerInteracting.current.x
        const deltaY = e.clientY - pointerInteracting.current.y
        pointerInteracting.current = { x: e.clientX, y: e.clientY }
        rPhi.set(rPhi.get() + deltaX / MOVEMENT_DAMPING)
        rTheta.set(rTheta.get() - deltaY / MOVEMENT_DAMPING)
      } else if (containerRef.current && config.markers) {
        const rect = containerRef.current.getBoundingClientRect()
        const x = (e.clientX - rect.left) / rect.width
        const y = (e.clientY - rect.top) / rect.height
        setTooltipPos({ x: e.clientX, y: e.clientY })
        const markers = config.markers as GlobeMarkerWithLabel[]
        if (markers.length > 0 && markers.some((m) => m.label)) {
          const phi = phiRef.current + rsPhi.get()
          const theta = DEFAULT_THETA + rsTheta.get()
          let best: { label: string; dist: number } | null = null
          for (const m of markers) {
            if (!m.label) continue
            const [lat, lng] = m.location
            const latR = (lat * Math.PI) / 180
            const lngR = (lng * Math.PI) / 180
            const cosLat = Math.cos(latR)
            const px = 0.5 + (Math.cos(phi) * Math.sin(lngR) * cosLat - Math.sin(phi) * Math.cos(lngR) * cosLat) * 0.4
            const py = 0.5 + (Math.sin(theta) * Math.cos(latR) + Math.cos(theta) * (Math.sin(phi) * Math.sin(lngR) * cosLat + Math.cos(phi) * Math.cos(lngR) * cosLat)) * 0.4
            const dist = (x - px) ** 2 + (y - py) ** 2
            if (!best || dist < best.dist) best = { label: m.label, dist }
          }
          setHoveredMarker(best && best.dist < 0.015 ? best.label : null)
        }
      }
    },
    [rPhi, rTheta, rsPhi, rsTheta, config.markers]
  )

  const handlePointerUp = useCallback(() => {
    updatePointerInteraction(null)
  }, [updatePointerInteraction])

  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      e.preventDefault()
      const delta = e.deltaY > 0 ? -0.08 : 0.08
      const next = Math.max(SCALE_MIN, Math.min(SCALE_MAX, scaleMotion.get() + delta))
      scaleMotion.set(next)
    },
    [scaleMotion]
  )

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const updateDimensions = () => {
      const w = canvas.offsetWidth
      const h = canvas.offsetHeight || w
      if (w > 0) {
        setDimensions((prev) => (prev.width === w && prev.height === h ? prev : { width: w, height: h }))
      }
    }
    updateDimensions()
    const ro = new ResizeObserver(updateDimensions)
    ro.observe(canvas)
    window.addEventListener("resize", updateDimensions)
    return () => {
      ro.disconnect()
      window.removeEventListener("resize", updateDimensions)
    }
  }, [])

  useEffect(() => {
    const { width } = dimensions
    if (!canvasRef.current || width === 0) return
    const globe = createGlobe(canvasRef.current, {
      ...config,
      width: width * 2,
      height: width * 2,
      scale: scaleSpring.get(),
      onRender: (state) => {
        if (!pointerInteracting.current) phiRef.current += 0.003
        phiRef.current = phiRef.current % (2 * Math.PI)
        const phi = phiRef.current + rsPhi.get()
        const theta = Math.max(-Math.PI / 2 + 0.1, Math.min(Math.PI / 2 - 0.1, DEFAULT_THETA + rsTheta.get()))
        state.phi = phi
        state.theta = theta
        state.scale = scaleSpring.get()
        state.width = width * 2
        state.height = width * 2
      },
    })
    const timer = setTimeout(() => {
      if (canvasRef.current) canvasRef.current.style.opacity = "1"
    }, 0)
    return () => {
      globe.destroy()
      clearTimeout(timer)
    }
  }, [rsPhi, rsTheta, scaleSpring, config, dimensions])

  return (
    <div
      ref={containerRef}
      className={cn("relative w-full h-full min-h-[400px] overflow-hidden", className)}
      onPointerLeave={() => setHoveredMarker(null)}
      onWheel={handleWheel}
    >
      <canvas
        ref={canvasRef}
        className="cursor-grab opacity-0 transition-opacity w-full h-full touch-none"
        style={{ aspectRatio: "1", display: "block" }}
        onPointerDown={handlePointerDown}
        onPointerUp={handlePointerUp}
        onPointerOut={handlePointerUp}
        onPointerMove={handlePointerMove}
        onContextMenu={(e) => e.preventDefault()}
      />
      {hoveredMarker && (
        <div
          className="pointer-events-none fixed z-50 max-w-[220px] rounded-md border border-border bg-popover px-3 py-2 text-sm text-popover-foreground shadow-md"
          style={{ left: tooltipPos.x + 12, top: tooltipPos.y + 8 }}
        >
          {hoveredMarker}
        </div>
      )}
    </div>
  )
}
