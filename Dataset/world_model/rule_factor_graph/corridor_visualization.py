"""Render the exported corridor factor graph as a two-slice SVG overview."""
from __future__ import annotations

import html
import json
from pathlib import Path


def render_graph(graph: dict) -> str:
    """Use exported identities and coefficients; the overview omits later slices."""
    ticks = sorted({node['typed_key']['tick'] for node in graph['nodes']
                    if node['node_type'] == 'variable'})
    if len(ticks) < 3:
        raise ValueError('The overview requires an anchor and two future slices')
    if len(graph['corridor_ids']) != 2:
        raise ValueError('The overview requires exactly two corridor owners')
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="920" viewBox="0 0 1400 920">',
             '<rect width="1400" height="920" fill="#f8fafc"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#172337}.title{font-size:25px;font-weight:bold}.body{font-size:15px}.small{font-size:12px}.label{font-size:16px;font-weight:bold}.edge{stroke:#718096;stroke-width:1.7;fill:none}</style>']

    def label(x, y, value, cls='body', anchor='start'):
        parts.append(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{html.escape(str(value))}</text>')

    def line(x1, y1, x2, y2):
        parts.append(f'<path d="M{x1},{y1} L{x2},{y2}" class="edge"/>')

    def box(x, y, w, h, value, fill, stroke='#718096'):
        parts.append(f'<rect x="{x-w/2}" y="{y-h/2}" width="{w}" height="{h}" rx="5" fill="{fill}" stroke="{stroke}"/>')
        label(x, y+5, value, 'label', 'middle')

    def variable(x, y, name):
        parts.append(f'<circle cx="{x}" cy="{y}" r="27" fill="#e1edff" stroke="#3267a0" stroke-width="2"/>')
        label(x, y+6, name, 'label', 'middle')

    label(40, 42, 'Shared-speed corridor prediction: domain-specific neural prototype', 'title')
    label(40, 70, graph['episode_id']+' | '+graph['actor_id'])
    label(40, 97, 'One sampled actor trajectory per branch; future B and I remain latent. Lines show factor incidence, not causation.')
    label(40, 125, 'Corridor 1: '+graph['corridor_ids'][0], 'small')
    label(40, 145, 'Corridor 2: '+graph['corridor_ids'][1], 'small')
    box(330, 185, 580, 38, 'Frozen history + shared forecast evidence', '#eef2f6')
    box(1040, 185, 600, 38, 'Fixed geometry + episode tau (parameters)', '#eef2f6')

    state_positions = []
    for col, tick in enumerate(ticks[1:3]):
        offset = col*690
        label(offset+345, 245, f'Future tick {tick}', 'label', 'middle')
        b, i1, i2 = (offset+345, 305), (offset+150, 435), (offset+540, 435)
        state_positions.append((b, i1, i2))
        for p, name in ((b,'B'), (i1,'I1'), (i2,'I2')):
            variable(*p, name)
        for x, y, name, target in ((offset+345,365,'B emission',b),
                                   (offset+150,500,'I emission',i1),
                                   (offset+540,500,'I emission',i2)):
            line(x,y-18,*target)
            box(x,y,128,34,name,'#d9f1df','#397b49')
        for x, target, name in ((offset+240,i1,'gamma1'),(offset+450,i2,'gamma2')):
            line(*b,x,390)
            line(x,390,*target)
            box(x,390,91,32,name,'#ffedcc','#a9721d')
        line(*i1,offset+345,465)
        line(*i2,offset+345,465)
        box(offset+345,465,164,32,'Geometry support','#e4e7ee','#475569')
        for corridor, inside, center in ((1,i1,offset+150),(2,i2,offset+540)):
            for index, (symbol, formula) in enumerate((('O','I'),('M','I & B'),('S','I & !B'))):
                x = center+(index-1)*88
                line(*inside,x,590)
                if symbol in ('M','S'):
                    line(*b,x,590)
                box(x,590,80,36,formula,'#e4e7ee','#475569')
                line(x,608,x,654)
                variable(x,682,symbol+str(corridor))
        label(offset+345, 741, 'O=I; M=I&B; S=I&!B. B emission occurs once.', 'small', 'middle')

    # Temporal scopes are listed separately to avoid drawing misleading crossovers.
    box(700, 797, 1270, 40, 'Temporal factors: lambda_B [B_t=B_t+1] + lambda_I1 [I1_t=I1_t+1] + lambda_I2 [I2_t=I2_t+1]', '#d9f1df','#397b49')
    label(40, 845, 'Blue circles: variables   Gray boxes: fixed support   Green: learned emissions/persistence   Amber: learned gamma association')
    label(40, 870, f'Overview: 2 of {len(ticks)-1} future slices. Full exported graph: {len(graph["nodes"])} nodes, {len(graph["edges"])} edges. Hard support is never gated.', 'small')
    label(40, 892, 'Manual B/I features and scopes; a unified typed-variable encoder and contract-generated neural scopes are not implemented.', 'small')
    parts.append('</svg>')
    return '\n'.join(parts)+'\n'


def main():
    directory = Path(__file__).resolve().parents[1]/'runs'/'rule_factor_graph_v3_corridor'
    graph = json.loads((directory/'graph.json').read_text())
    (directory/'graph.svg').write_text(render_graph(graph))
    print(directory/'graph.svg')


if __name__ == '__main__':
    main()
