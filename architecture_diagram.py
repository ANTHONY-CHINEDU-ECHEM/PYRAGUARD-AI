"""
Generate a professional PyraGuard AI architecture flowchart diagram.
This script creates a high-quality SVG visualization of the system pipeline.
"""

import os
from pathlib import Path


def generate_architecture_diagram():
    """Generate a professional architecture flowchart."""
    
    svg_content = '''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1400 900" width="1400" height="900">
  <defs>
    <style>
      .stage-box { stroke: #333; stroke-width: 2; }
      .vision-box { fill: #E3F2FD; }
      .temporal-box { fill: #F3E5F5; }
      .hazard-box { fill: #FCE4EC; }
      .rag-box { fill: #E8F5E9; }
      .action-box { fill: #E1F5FE; }
      .text-title { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; font-weight: 700; font-size: 16px; fill: #0D47A1; }
      .text-content { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; font-size: 13px; fill: #333; }
      .text-label { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; font-weight: 600; font-size: 12px; fill: #555; }
      .arrow { fill: none; stroke: #666; stroke-width: 2.5; marker-end: url(#arrowhead); }
      .feedback-arrow { fill: none; stroke: #999; stroke-width: 2; stroke-dasharray: 5,5; marker-end: url(#arrowhead-light); }
    </style>
    <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
      <polygon points="0 0, 10 3, 0 6" fill="#666" />
    </marker>
    <marker id="arrowhead-light" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
      <polygon points="0 0, 10 3, 0 6" fill="#999" />
    </marker>
  </defs>

  <!-- Title -->
  <text x="700" y="35" text-anchor="middle" style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; font-weight: 700; font-size: 24px; fill: #1a1a1a;">
    PyraGuard AI: From Pixels to Cited Response Plan
  </text>

  <!-- ========== VISION PIPELINE ROW ========== -->
  
  <!-- Stage 1: Sensing -->
  <rect x="20" y="80" width="200" height="140" rx="8" class="stage-box vision-box"/>
  <text x="120" y="105" text-anchor="middle" class="text-title">1. Sensing</text>
  <text x="30" y="130" class="text-content">• RGB cameras</text>
  <text x="30" y="150" class="text-content">• Thermal cameras</text>
  <text x="30" y="170" class="text-content">• Video files</text>
  <text x="30" y="190" class="text-content">• Generated clips</text>
  <text x="30" y="210" class="text-content">• RTSP streams</text>

  <!-- Arrow 1 → 2 -->
  <path d="M 220 150 L 280 150" class="arrow"/>

  <!-- Stage 2: Vision -->
  <rect x="280" y="80" width="220" height="140" rx="8" class="stage-box vision-box"/>
  <text x="390" y="105" text-anchor="middle" class="text-title">2. Vision</text>
  <text x="290" y="130" class="text-content">• YOLO fire & smoke</text>
  <text x="290" y="150" class="text-content">• Classical colour detection</text>
  <text x="290" y="170" class="text-content">• Motion detector</text>
  <text x="290" y="190" class="text-content">• Thermal hot spots</text>
  <text x="290" y="210" class="text-content">• Ensemble fusion</text>

  <!-- Arrow 2 → 3 -->
  <path d="M 500 150 L 560 150" class="arrow"/>

  <!-- Stage 3: Temporal -->
  <rect x="560" y="80" width="220" height="140" rx="8" class="stage-box temporal-box"/>
  <text x="670" y="105" text-anchor="middle" class="text-title">3. Temporal</text>
  <text x="570" y="130" class="text-content">• IoU tracking</text>
  <text x="570" y="150" class="text-content">• Growth rate analysis</text>
  <text x="570" y="170" class="text-content">• Flicker detection</text>
  <text x="570" y="190" class="text-content">• Drift & rise tracking</text>
  <text x="570" y="210" class="text-content">• Persistence scoring</text>

  <!-- Arrow 3 → 4 -->
  <path d="M 780 150 L 840 150" class="arrow"/>

  <!-- Stage 4: Hazard -->
  <rect x="840" y="80" width="220" height="140" rx="8" class="stage-box hazard-box"/>
  <text x="950" y="105" text-anchor="middle" class="text-title">4. Hazard Scoring</text>
  <text x="850" y="130" class="text-content">• Score 0–100</text>
  <text x="850" y="150" class="text-content">• 5 severity levels</text>
  <text x="850" y="170" class="text-content">• Time-based confirmation</text>
  <text x="850" y="190" class="text-content">• Incident ratchet</text>
  <text x="850" y="210" class="text-content">• Escalation logic</text>

  <!-- Arrow 4 → 5 -->
  <path d="M 1060 150 L 1120 150" class="arrow"/>

  <!-- Stage 5: Scene Classification -->
  <rect x="1120" y="80" width="240" height="140" rx="8" class="stage-box hazard-box"/>
  <text x="1240" y="105" text-anchor="middle" class="text-title">5. Scene Context</text>
  <text x="1130" y="130" class="text-content">• Zone type classification</text>
  <text x="1130" y="150" class="text-content">• Materials & contents</text>
  <text x="1130" y="170" class="text-content">• Occupancy level</text>
  <text x="1130" y="190" class="text-content">• Building layout labels</text>
  <text x="1130" y="210" class="text-content">• Risk escalation</text>

  <!-- ========== RAG PIPELINE ROW ========== -->

  <!-- Knowledge Base -->
  <rect x="20" y="300" width="220" height="160" rx="8" class="stage-box rag-box"/>
  <text x="130" y="330" text-anchor="middle" class="text-title">Knowledge Base</text>
  <text x="30" y="360" class="text-content">• 18 guidance documents</text>
  <text x="30" y="380" class="text-content">• 81 semantic chunks</text>
  <text x="30" y="400" class="text-content">  – 23 action chunks</text>
  <text x="30" y="420" class="text-content">  – 17 prohibition chunks</text>
  <text x="30" y="440" class="text-content">• Reference image library</text>

  <!-- Arrow: Hazard Score → Retrieval -->
  <path d="M 950 220 Q 950 260 610 280" class="arrow"/>
  <text x="880" y="245" class="text-label">incident state change</text>

  <!-- Stage 6: Multimodal Retrieval -->
  <rect x="280" y="300" width="240" height="160" rx="8" class="stage-box rag-box"/>
  <text x="400" y="330" text-anchor="middle" class="text-title">6. Multimodal Retrieval</text>
  <text x="290" y="360" class="text-content">• Query decomposition</text>
  <text x="290" y="380" class="text-content">• Dense + BM25 fusion</text>
  <text x="290" y="400" class="text-content">• Rank fusion (reciprocal)</text>
  <text x="290" y="420" class="text-content">• Hazard & zone boosting</text>
  <text x="290" y="440" class="text-content">• Level filtering & images</text>

  <!-- Arrow: KB → Retrieval -->
  <path d="M 240 380 L 280 380" class="arrow"/>

  <!-- Arrow: Retrieval → Generation -->
  <path d="M 520 380 L 580 380" class="arrow"/>

  <!-- Stage 7: Grounded Generation -->
  <rect x="580" y="300" width="240" height="160" rx="8" class="stage-box rag-box"/>
  <text x="700" y="330" text-anchor="middle" class="text-title">7. Grounded Generation</text>
  <text x="590" y="360" class="text-content">• Extractive or LLM plans</text>
  <text x="590" y="380" class="text-content">• Citation per line</text>
  <text x="590" y="400" class="text-content">• Support verification</text>
  <text x="590" y="420" class="text-content">• Stage-based gating</text>
  <text x="590" y="440" class="text-content">• Fallback on failure</text>

  <!-- Arrow: Generation → Action -->
  <path d="M 820 380 L 880 380" class="arrow"/>

  <!-- Stage 8: Action & Delivery -->
  <rect x="880" y="300" width="240" height="160" rx="8" class="stage-box action-box"/>
  <text x="1000" y="330" text-anchor="middle" class="text-title">8. Action &amp; Delivery</text>
  <text x="890" y="360" class="text-content">• Response plan + sources</text>
  <text x="890" y="380" class="text-content">• Evacuation routes</text>
  <text x="890" y="400" class="text-content">• Multi-channel alerts</text>
  <text x="890" y="420" class="text-content">  – Console, JSON, webhook</text>
  <text x="890" y="440" class="text-content">  – Email, API, dashboard</text>

  <!-- ========== EVACUATION FLOW ========== -->

  <!-- Evacuation Module (separate visual element) -->
  <rect x="1180" y="300" width="180" height="160" rx="8" style="fill: #FFF3E0; stroke: #333; stroke-width: 2;"/>
  <text x="1270" y="330" text-anchor="middle" class="text-title">Evacuation Router</text>
  <text x="1190" y="360" class="text-content">• Site graph model</text>
  <text x="1190" y="380" class="text-content">• Hazard-aware routing</text>
  <text x="1190" y="400" class="text-content">• Refuge locations</text>
  <text x="1190" y="420" class="text-content">• Blocked zone handling</text>
  <text x="1190" y="440" class="text-content">• Trapped zone alerts</text>

  <!-- Arrow: Hazard → Evacuation -->
  <path d="M 1060 220 L 1200 300" class="arrow"/>

  <!-- ========== FEEDBACK LOOP ========== -->
  <!-- Feedback from Action back to Hazard (incident state change) -->
  <path d="M 1060 300 Q 1060 260 1060 220" class="feedback-arrow"/>

  <!-- ========== LEGEND / KEY ========== -->
  <g id="legend">
    <text x="20" y="520" style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; font-weight: 600; font-size: 13px; fill: #333;">
      Key Capabilities:
    </text>
    <text x="20" y="545" class="text-label">✓ Detects small flames (&lt;0.3% frame)</text>
    <text x="20" y="565" class="text-label">✓ Confirms before alarming (time-based)</text>
    <text x="20" y="585" class="text-label">✓ Grades hazard (0–100 score, 5 levels)</text>
    <text x="20" y="605" class="text-label">✓ Retrieves cited guidance automatically</text>
    <text x="20" y="625" class="text-label">✓ Routes evacuations hazard-aware</text>
    <text x="20" y="645" class="text-label">✓ Every plan line is cited with sources</text>

    <text x="700" y="545" class="text-label">✓ Adapts advice per incident stage</text>
    <text x="700" y="565" class="text-label">✓ ~80% false alarm reduction</text>
    <text x="700" y="585" class="text-label">✓ Runs on single CPU core (18ms/frame)</text>
    <text x="700" y="605" class="text-label">✓ No GPU required · No cloud dependency</text>
    <text x="700" y="625" class="text-label">✓ Full audit trail & transparency</text>
    <text x="700" y="645" class="text-label">✓ Integrated FastAPI + Streamlit delivery</text>
  </g>

  <!-- ========== FOOTER ========== -->
  <text x="700" y="710" text-anchor="middle" style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; font-size: 12px; fill: #666;">
    End-to-end autonomous early-stage thermal hazard detection with cited response planning
  </text>

  <!-- Color legend boxes -->
  <rect x="20" y="740" width="12" height="12" rx="2" class="vision-box" style="stroke: #333; stroke-width: 1;"/>
  <text x="40" y="750" style="font-size: 12px; fill: #666; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;">Vision &amp; Temporal Detection</text>

  <rect x="270" y="740" width="12" height="12" rx="2" class="hazard-box" style="stroke: #333; stroke-width: 1;"/>
  <text x="290" y="750" style="font-size: 12px; fill: #666; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;">Hazard Scoring &amp; Scene Context</text>

  <rect x="550" y="740" width="12" height="12" rx="2" class="rag-box" style="stroke: #333; stroke-width: 1;"/>
  <text x="570" y="750" style="font-size: 12px; fill: #666; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;">Knowledge &amp; Retrieval-Augmented Generation</text>

  <rect x="1000" y="740" width="12" height="12" rx="2" class="action-box" style="stroke: #333; stroke-width: 1;"/>
  <text x="1020" y="750" style="font-size: 12px; fill: #666; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;">Action &amp; Output Delivery</text>

</svg>
'''

    return svg_content


def save_diagram():
    """Save the diagram to an SVG file and also create a high-quality PNG export guide."""
    
    # Generate SVG
    svg = generate_architecture_diagram()
    
    # Save as SVG
    svg_path = Path("architecture_diagram.svg")
    with open(svg_path, "w") as f:
        f.write(svg)
    
    print(f"✓ Created: {svg_path}")
    print("\nTo convert to PNG with high quality, use one of these methods:\n")
    
    print("1. Using ImageMagick (recommended for quality):")
    print("   convert -density 300 -background white architecture_diagram.svg architecture_diagram.png\n")
    
    print("2. Using Inkscape (best for precise rendering):")
    print("   inkscape architecture_diagram.svg --export-type=png --export-dpi=300 -o architecture_diagram.png\n")
    
    print("3. Using cairosvg (if installed):")
    print("   cairosvg architecture_diagram.svg -o architecture_diagram.png -d 300\n")
    
    print("4. Online: https://cloudconvert.com or https://svg2png.com\n")


if __name__ == "__main__":
    save_diagram()
