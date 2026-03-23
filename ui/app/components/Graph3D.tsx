"use client";

import { useRef, useState, useEffect, useMemo, useCallback } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Html } from "@react-three/drei";
import * as THREE from "three";
import {
  forceSimulation,
  forceLink,
  forceManyBody,
  forceCenter,
} from "d3-force-3d";
import { useMemory } from "../context";
import { getCategoryColor, EDGE_COLORS, type MemoryNode } from "../types";

// ─── Helpers ────────────────────────────────────────────────────────────────

function hashString(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = (Math.imul(31, h) + s.charCodeAt(i)) | 0;
  }
  return h;
}

function hexToRGB(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) / 255, ((n >> 8) & 0xff) / 255, (n & 0xff) / 255];
}

const _tempVec3 = new THREE.Vector3();

// ─── Scene content ──────────────────────────────────────────────────────────

interface SimNode {
  id: string;
  x: number;
  y: number;
  z: number;
  vx?: number;
  vy?: number;
  vz?: number;
  node: MemoryNode;
}

function GraphScene() {
  const {
    nodes,
    edges,
    selectedNodeId,
    setSelectedNodeId,
    highlightedNodeIds,
    newNodeIds,
  } = useMemory();

  const [hoveredId, setHoveredId] = useState<string | null>(null);

  // Refs for imperative updates
  const nodeRefs = useRef<Map<string, THREE.Mesh>>(new Map());
  const edgeRef = useRef<THREE.LineSegments>(null);
  const simNodesRef = useRef<Map<string, SimNode>>(new Map());
  const newScaleRef = useRef<Map<string, number>>(new Map());

  // ── Force layout ──────────────────────────────────────────────────────────
  // Compute once when data changes, store base positions
  const basePositions = useMemo(() => {
    if (nodes.length === 0) return new Map<string, [number, number, number]>();

    const simNodes: SimNode[] = nodes.map((n) => ({
      id: n.id,
      x: (Math.random() - 0.5) * 40,
      y: (Math.random() - 0.5) * 40,
      z: (Math.random() - 0.5) * 40,
      node: n,
    }));

    const simLinks = edges
      .filter((e) => {
        const hasSource = nodes.some((n) => n.id === e.source);
        const hasTarget = nodes.some((n) => n.id === e.target);
        return hasSource && hasTarget;
      })
      .map((e) => ({ source: e.source, target: e.target }));

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const sim = forceSimulation(simNodes as any, 3)
      .force("charge", forceManyBody().strength(-60))
      .force(
        "link",
        forceLink(simLinks)
          .id((d: any) => d.id) // eslint-disable-line @typescript-eslint/no-explicit-any
          .distance(12)
          .strength(0.5)
      )
      .force("center", forceCenter());

    sim.stop();
    for (let i = 0; i < 300; i++) sim.tick();

    const posMap = new Map<string, [number, number, number]>();
    const nodeMap = new Map<string, SimNode>();
    simNodes.forEach((n) => {
      posMap.set(n.id, [n.x ?? 0, n.y ?? 0, n.z ?? 0]);
      nodeMap.set(n.id, n);
    });
    simNodesRef.current = nodeMap;
    return posMap;
  }, [nodes, edges]);

  // ── Edge geometry data ────────────────────────────────────────────────────
  const validEdges = useMemo(
    () =>
      edges.filter(
        (e) => basePositions.has(e.source) && basePositions.has(e.target)
      ),
    [edges, basePositions]
  );

  const edgeColorArray = useMemo(() => {
    const arr = new Float32Array(validEdges.length * 6);
    validEdges.forEach((edge, i) => {
      const hex = EDGE_COLORS[edge.type] ?? "#6b7280";
      const [r, g, b] = hexToRGB(hex);
      arr[i * 6] = r;
      arr[i * 6 + 1] = g;
      arr[i * 6 + 2] = b;
      arr[i * 6 + 3] = r;
      arr[i * 6 + 4] = g;
      arr[i * 6 + 5] = b;
    });
    return arr;
  }, [validEdges]);

  // ── Camera fit ────────────────────────────────────────────────────────────
  const { camera } = useThree();
  useEffect(() => {
    if (basePositions.size === 0) return;
    let maxDist = 0;
    basePositions.forEach(([x, y, z]) => {
      const d = Math.sqrt(x * x + y * y + z * z);
      if (d > maxDist) maxDist = d;
    });
    const dist = Math.max(maxDist * 2.5, 30);
    camera.position.set(dist * 0.8, dist * 0.6, dist * 0.8);
    camera.lookAt(0, 0, 0);
  }, [basePositions, camera]);

  // ── Per-frame animation ───────────────────────────────────────────────────
  useFrame(({ clock }) => {
    const t = clock.elapsedTime;

    // Update each node mesh
    nodeRefs.current.forEach((mesh, id) => {
      const base = basePositions.get(id);
      if (!base) return;

      // Gentle floating
      const h = hashString(id);
      const floatY = Math.sin(t * 0.4 + h * 0.01) * 0.25;
      const floatX = Math.cos(t * 0.3 + h * 0.013) * 0.12;
      const floatZ = Math.sin(t * 0.35 + h * 0.017) * 0.1;
      mesh.position.set(
        base[0] + floatX,
        base[1] + floatY,
        base[2] + floatZ
      );

      // Scale: new-node spring animation, or hover/select/highlight
      const isNew = newNodeIds.has(id);
      if (isNew) {
        const progress = newScaleRef.current.get(id) ?? 0;
        const next = Math.min(progress + 0.02, 1);
        newScaleRef.current.set(id, next);
        const s = easeOutBack(next);
        mesh.scale.setScalar(s);
      } else {
        const isHovered = hoveredId === id;
        const isSelected = selectedNodeId === id;
        const isHighlighted = highlightedNodeIds.has(id);
        const target = isHovered
          ? 1.5
          : isSelected
            ? 1.35
            : isHighlighted
              ? 1.25
              : 1;
        mesh.scale.lerp(_tempVec3.set(target, target, target), 0.12);
      }

      // Emissive pulse for highlighted nodes
      const mat = mesh.material as THREE.MeshStandardMaterial;
      if (mat) {
        const isHighlighted = highlightedNodeIds.has(id);
        const isSelected = selectedNodeId === id;
        const node = simNodesRef.current.get(id)?.node;
        const baseEmissive = node?.isCurrent ? 0.4 : 0.08;
        let targetEmissive = baseEmissive;
        if (isSelected) targetEmissive = 1.2;
        else if (isHighlighted)
          targetEmissive = 0.8 + Math.sin(t * 3) * 0.3;
        else if (hoveredId === id) targetEmissive = 0.9;
        mat.emissiveIntensity += (targetEmissive - mat.emissiveIntensity) * 0.1;
      }
    });

    // Update edge line positions
    if (edgeRef.current && validEdges.length > 0) {
      const geom = edgeRef.current.geometry;
      const posAttr = geom.getAttribute("position") as THREE.BufferAttribute;
      if (!posAttr || posAttr.count !== validEdges.length * 2) {
        const positions = new Float32Array(validEdges.length * 6);
        geom.setAttribute(
          "position",
          new THREE.BufferAttribute(positions, 3)
        );
        geom.setAttribute(
          "color",
          new THREE.BufferAttribute(edgeColorArray, 3)
        );
      }
      const arr = (geom.getAttribute("position") as THREE.BufferAttribute)
        .array as Float32Array;
      validEdges.forEach((edge, i) => {
        const sMesh = nodeRefs.current.get(edge.source);
        const tMesh = nodeRefs.current.get(edge.target);
        if (sMesh && tMesh) {
          arr[i * 6] = sMesh.position.x;
          arr[i * 6 + 1] = sMesh.position.y;
          arr[i * 6 + 2] = sMesh.position.z;
          arr[i * 6 + 3] = tMesh.position.x;
          arr[i * 6 + 4] = tMesh.position.y;
          arr[i * 6 + 5] = tMesh.position.z;
        }
      });
      (geom.getAttribute("position") as THREE.BufferAttribute).needsUpdate =
        true;
    }
  });

  // ── Hovered node tooltip ──────────────────────────────────────────────────
  const hoveredNode = hoveredId ? nodes.find((n) => n.id === hoveredId) : null;
  const hoveredMesh = hoveredId ? nodeRefs.current.get(hoveredId) : null;

  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.35} />
      <pointLight position={[60, 60, 60]} intensity={0.8} color="#e0e7ff" />
      <pointLight position={[-40, -30, -50]} intensity={0.3} color="#818cf8" />

      {/* Background particles — subtle depth */}
      <BackgroundParticles />

      {/* Edges */}
      <lineSegments ref={edgeRef} frustumCulled={false}>
        <bufferGeometry />
        <lineBasicMaterial vertexColors transparent opacity={0.25} />
      </lineSegments>

      {/* Nodes */}
      {nodes.map((node) => {
        const pos = basePositions.get(node.id);
        if (!pos) return null;
        const color = getCategoryColor(node.category);
        const confidence = node.confidence ?? 1;
        const radius = node.isCurrent
          ? 0.35 + confidence * 0.25
          : 0.2 + confidence * 0.1;

        return (
          <mesh
            key={node.id}
            position={pos}
            ref={(el) => {
              if (el) nodeRefs.current.set(node.id, el);
              return () => {
                nodeRefs.current.delete(node.id);
              };
            }}
            scale={newNodeIds.has(node.id) ? 0.01 : 1}
            onPointerOver={(e) => {
              e.stopPropagation();
              setHoveredId(node.id);
              document.body.style.cursor = "pointer";
            }}
            onPointerOut={() => {
              setHoveredId(null);
              document.body.style.cursor = "auto";
            }}
            onClick={(e) => {
              e.stopPropagation();
              setSelectedNodeId(node.id);
            }}
          >
            <sphereGeometry args={[radius, 20, 20]} />
            <meshStandardMaterial
              color={color}
              emissive={color}
              emissiveIntensity={node.isCurrent ? 0.4 : 0.08}
              transparent
              opacity={node.isCurrent ? 0.95 : 0.25}
              roughness={0.3}
              metalness={0.6}
            />
          </mesh>
        );
      })}

      {/* Hover tooltip */}
      {hoveredNode && hoveredMesh && (
        <Html
          position={[
            hoveredMesh.position.x,
            hoveredMesh.position.y + 1.2,
            hoveredMesh.position.z,
          ]}
          center
          distanceFactor={30}
          style={{ pointerEvents: "none" }}
        >
          <div className="bg-zinc-900/95 border border-zinc-700/50 rounded-lg px-3 py-2 max-w-[220px] shadow-xl backdrop-blur-sm">
            <div className="flex items-center gap-1.5 mb-1">
              <span
                className="w-1.5 h-1.5 rounded-full"
                style={{
                  backgroundColor: getCategoryColor(hoveredNode.category),
                }}
              />
              <span className="text-[9px] text-zinc-500 uppercase font-medium">
                {hoveredNode.category}
              </span>
              {!hoveredNode.isCurrent && (
                <span className="text-[9px] text-zinc-600 ml-auto">
                  superseded
                </span>
              )}
            </div>
            <p className="text-[11px] text-zinc-200 leading-snug">
              {hoveredNode.content.length > 100
                ? hoveredNode.content.slice(0, 100) + "..."
                : hoveredNode.content}
            </p>
          </div>
        </Html>
      )}

      {/* Controls — enhanced for smooth panning/zoom */}
      <EnhancedControls selectedNodeId={selectedNodeId} basePositions={basePositions} nodeRefs={nodeRefs} />
    </>
  );
}

// ─── Background particles for depth ─────────────────────────────────────────

function BackgroundParticles() {
  const ref = useRef<THREE.Points>(null);
  const count = 200;

  const positions = useMemo(() => {
    const arr = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      arr[i * 3] = (Math.random() - 0.5) * 150;
      arr[i * 3 + 1] = (Math.random() - 0.5) * 150;
      arr[i * 3 + 2] = (Math.random() - 0.5) * 150;
    }
    return arr;
  }, []);

  useFrame(({ clock }) => {
    if (ref.current) {
      ref.current.rotation.y = clock.elapsedTime * 0.01;
      ref.current.rotation.x = clock.elapsedTime * 0.005;
    }
  });

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.15}
        color="#4b5563"
        transparent
        opacity={0.4}
        sizeAttenuation
      />
    </points>
  );
}

// ─── Enhanced Controls ──────────────────────────────────────────────────────

function EnhancedControls({
  selectedNodeId,
  basePositions,
  nodeRefs,
}: {
  selectedNodeId: string | null;
  basePositions: Map<string, [number, number, number]>;
  nodeRefs: React.RefObject<Map<string, THREE.Mesh>>;
}) {
  const controlsRef = useRef<any>(null);
  const { camera } = useThree();
  const targetRef = useRef(new THREE.Vector3(0, 0, 0));
  const flyingRef = useRef(false);

  // Fly to selected node on double-click or selection change
  useEffect(() => {
    if (!selectedNodeId || !controlsRef.current) return;
    const mesh = nodeRefs.current?.get(selectedNodeId);
    if (!mesh) return;

    const nodePos = mesh.position.clone();
    const controls = controlsRef.current;

    // Animate camera to focus on selected node
    flyingRef.current = true;
    const startTarget = controls.target.clone();
    const startPos = camera.position.clone();

    // Position camera at a nice distance from the node
    const offset = camera.position.clone().sub(controls.target).normalize().multiplyScalar(15);
    const endPos = nodePos.clone().add(offset);

    let progress = 0;
    const animate = () => {
      progress += 0.035;
      if (progress >= 1) {
        controls.target.copy(nodePos);
        camera.position.copy(endPos);
        flyingRef.current = false;
        return;
      }
      const t = easeInOutCubic(progress);
      controls.target.lerpVectors(startTarget, nodePos, t);
      camera.position.lerpVectors(startPos, endPos, t);
      requestAnimationFrame(animate);
    };
    animate();
  }, [selectedNodeId, camera, basePositions, nodeRefs]);

  return (
    <OrbitControls
      ref={controlsRef}
      enableDamping
      dampingFactor={0.12}
      rotateSpeed={0.8}
      zoomSpeed={1.4}
      panSpeed={1.2}
      minDistance={3}
      maxDistance={300}
      enablePan={true}
      mouseButtons={{
        LEFT: THREE.MOUSE.ROTATE,
        MIDDLE: THREE.MOUSE.DOLLY,
        RIGHT: THREE.MOUSE.PAN,
      }}
      touches={{
        ONE: THREE.TOUCH.ROTATE,
        TWO: THREE.TOUCH.DOLLY_PAN,
      }}
      // Smooth zoom
      zoomToCursor={true}
    />
  );
}

function easeInOutCubic(x: number): number {
  return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
}

// ─── Controls Help Overlay ──────────────────────────────────────────────────

function ControlsHelp() {
  const [visible, setVisible] = useState(true);
  const [fading, setFading] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => {
      setFading(true);
      setTimeout(() => setVisible(false), 600);
    }, 6000);
    return () => clearTimeout(timer);
  }, []);

  if (!visible) return null;

  return (
    <div
      className={`absolute bottom-4 left-1/2 -translate-x-1/2 z-10 transition-opacity duration-500 ${fading ? "opacity-0" : "opacity-100"}`}
    >
      <div className="bg-zinc-900/90 border border-zinc-700/40 rounded-xl px-5 py-3 backdrop-blur-md shadow-2xl">
        <div className="flex items-center gap-6 text-[11px] text-zinc-400">
          <div className="flex items-center gap-1.5">
            <kbd className="bg-zinc-800 px-1.5 py-0.5 rounded text-zinc-300 font-mono text-[10px]">Left drag</kbd>
            <span>Rotate</span>
          </div>
          <div className="flex items-center gap-1.5">
            <kbd className="bg-zinc-800 px-1.5 py-0.5 rounded text-zinc-300 font-mono text-[10px]">Right drag</kbd>
            <span>Pan</span>
          </div>
          <div className="flex items-center gap-1.5">
            <kbd className="bg-zinc-800 px-1.5 py-0.5 rounded text-zinc-300 font-mono text-[10px]">Scroll</kbd>
            <span>Zoom</span>
          </div>
          <div className="flex items-center gap-1.5">
            <kbd className="bg-zinc-800 px-1.5 py-0.5 rounded text-zinc-300 font-mono text-[10px]">Click node</kbd>
            <span>Focus</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Zoom-to-Fit Button ─────────────────────────────────────────────────────

function ZoomControls({ onFit }: { onFit: () => void }) {
  return (
    <div className="absolute bottom-4 right-4 z-10 flex flex-col gap-1.5">
      <button
        onClick={onFit}
        title="Zoom to fit all nodes"
        className="bg-zinc-800/80 hover:bg-zinc-700/90 border border-zinc-700/40 rounded-lg w-8 h-8 flex items-center justify-center text-zinc-400 hover:text-zinc-200 transition-colors backdrop-blur-sm shadow-lg"
      >
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
          <rect x="2" y="2" width="12" height="12" rx="1.5" />
          <path d="M5 8h6M8 5v6" />
        </svg>
      </button>
    </div>
  );
}

// ─── Easing ─────────────────────────────────────────────────────────────────

function easeOutBack(x: number): number {
  const c1 = 1.70158;
  const c3 = c1 + 1;
  return 1 + c3 * Math.pow(x - 1, 3) + c1 * Math.pow(x - 1, 2);
}

// ─── Canvas wrapper ─────────────────────────────────────────────────────────

export default function Graph3D() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const handleZoomToFit = useCallback(() => {
    // Dispatch a custom event that the scene can listen for
    window.dispatchEvent(new CustomEvent("memory-graph-fit"));
  }, []);

  return (
    <div className="w-full h-full relative">
      <Canvas
        ref={canvasRef}
        camera={{ position: [50, 30, 50], fov: 55, near: 0.1, far: 500 }}
        gl={{ antialias: true, alpha: false }}
        onCreated={({ gl }) => {
          gl.setClearColor("#09090b");
        }}
        style={{ background: "#09090b" }}
      >
        <GraphScene />
        <FitListener />
      </Canvas>
      <ControlsHelp />
      <ZoomControls onFit={handleZoomToFit} />
    </div>
  );
}

/** Listens for the zoom-to-fit custom event inside the Canvas */
function FitListener() {
  const { camera } = useThree();
  const { nodes } = useMemory();

  useEffect(() => {
    const handler = () => {
      if (nodes.length === 0) return;
      // Compute bounding sphere
      let cx = 0, cy = 0, cz = 0;
      nodes.forEach((n) => {
        cx += (n as any).x ?? 0;
        cy += (n as any).y ?? 0;
        cz += (n as any).z ?? 0;
      });
      // Reset camera to default overview
      const dist = Math.max(nodes.length * 1.2, 40);
      camera.position.set(dist * 0.8, dist * 0.6, dist * 0.8);
      camera.lookAt(0, 0, 0);
    };
    window.addEventListener("memory-graph-fit", handler);
    return () => window.removeEventListener("memory-graph-fit", handler);
  }, [camera, nodes]);

  return null;
}
