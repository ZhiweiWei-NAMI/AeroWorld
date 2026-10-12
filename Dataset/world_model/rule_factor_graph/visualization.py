"""Render the two real oracle slices as a standalone native SVG diagram."""
from __future__ import annotations

from html import escape
from pathlib import Path


def write_case_svg(case: dict, output: Path) -> None:
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="830" viewBox="0 0 1280 830">',
             '<style>text{font-family:Arial,sans-serif;fill:#172b4d;font-size:15px}.title{font-size:23px;font-weight:bold}.small{font-size:12px}.edge{stroke:#334e68;stroke-width:2;fill:none}.evidence{stroke:#718096;stroke-width:1.4;stroke-dasharray:5 4;fill:none}</style>',
             '<rect width="1280" height="830" fill="#f7fafc"/>',
             '<text x="30" y="35" class="title">Corridor rule factors · real oracle replay</text>',
             '<text x="30" y="62">Hard incidence follows typed role, field owner and time joins. Dashed lines cite evidence.</text>']
    def box(x, y, w, h, fill, label, subtitle=''):
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{fill}" stroke="#718096"/>')
        parts.append(f'<text x="{x+12}" y="{y+25}">{escape(label)}</text>')
        if subtitle:
            parts.append(f'<text x="{x+12}" y="{y+46}" class="small">{escape(subtitle)}</text>')
    tau = case['summary']['governed_parameters']['aircraft_stationary_speed_threshold_mps']
    for offset,tick in enumerate((190,195)):
        top = 100 + offset*315
        record = case['records'][tick]
        values = record['record']['values']
        parts.append(f'<text x="30" y="{top}" class="title">tick {tick} · {tick/10:.1f} s</text>')
        # All endpoint geometry is explicit; line drawing carries no semantic inference.
        for sy,fy in ((top+60,top+40),(top+60,top+110),(top+60,top+180),(top+130,top+110),(top+130,top+180),(top+200,top+110),(top+200,top+180)):
            parts.append(f'<path d="M 600 {sy} L 730 {fy}" class="edge"/>')
        for fy in (top+40,top+110,top+180):
            parts.append(f'<path d="M 940 {fy} L 1030 {fy}" class="edge"/>')
        parts.append(f'<path d="M 285 {top+100} L 340 {top+60} M 285 {top+100} L 340 {top+130}" class="evidence"/>')
        parts.append(f'<path d="M 285 {top+215} L 340 {top+200}" class="evidence"/>')
        box(30,top+35,255,115,'#edf2f7','L0 state evidence',f'domain_state_observations.jsonl:{record["source_line"]}')
        parts.append(f'<text x="42" y="{top+111}" class="small">Exact actor + corridor IDs; tick {tick}</text>')
        box(30,top+180,255,65,'#edf2f7','Episode parameter evidence','world_truth_summary.json')
        box(340,top+30,260,60,'#d8effc',f'I = {str(values["aircraft_inside_corridor"]).lower()}','owner: actor + corridor')
        box(340,top+100,260,60,'#d8effc',f'v = {values["speed_mps"]:.6g} m/s','owner: actor; shared by M and S')
        box(340,top+170,260,60,'#e9d8fd',f'tau = {tau:g} m/s','episode parameter')
        truth=(values['aircraft_inside_corridor'],values['aircraft_inside_corridor'] and values['speed_mps']>tau,values['aircraft_inside_corridor'] and values['speed_mps']<=tau)
        for i,(formula,name,val) in enumerate(zip(('O = I','M = I AND v > tau','S = I AND v <= tau'),('O · occupies','M · moving','S · stationary'),truth)):
            box(730,top+10+70*i,210,60,'#feebc8',formula,'hard rule factor')
            box(1030,top+10+70*i,215,60,'#c6f6d5',name,f'predicate = {str(val).lower()}')
    parts.extend([f'<text x="30" y="752" class="small">actor: {escape(case["actor_id"])} · world:UnmannedAircraft</text>',
                  f'<text x="30" y="774" class="small">corridor: {escape(case["corridor_id"])} · world:AirspaceCorridor</text>',
                  '<text x="30" y="804" class="small">Oracle inputs fix one joint path. Learned factors, learned edges and predictive geometry sampling remain unimplemented.</text>', '</svg>'])
    output.write_text('\n'.join(parts)+'\n')
