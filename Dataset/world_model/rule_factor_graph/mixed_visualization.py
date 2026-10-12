"""Draw the fitted association scopes; full time/branch data stay in graph.json."""
from __future__ import annotations

import json
from html import escape
from pathlib import Path


def write_mixed_svg(output: Path) -> None:
    graph = json.loads((output / 'graph.json').read_text())
    result = json.loads((output / 'results.json').read_text())
    coefficients = graph['coefficient_summary']
    selected = graph['selected_structure']
    active = {'AL': selected in ('AL', 'AL_AP'), 'AP': selected in ('AP', 'AL_AP')}
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1220" height="870" viewBox="0 0 1220 870">',
             '<style>text{font:15px Arial,sans-serif;fill:#19324a}.title{font-size:23px;font-weight:bold}.small{font-size:12px}.hard{stroke:#26384c;stroke-width:2.4;fill:none}.learned{stroke:#087f8c;stroke-width:2.3;fill:none}.off{stroke:#94a3b8;stroke-width:2;stroke-dasharray:6 5;fill:none}.context{stroke:#64748b;stroke-width:1.5;stroke-dasharray:3 4;fill:none}</style>',
             '<rect width="1220" height="870" fill="#f7fafc"/>',
             '<text x="28" y="36" class="title">Communication joint prediction · fitted factor scopes</text>',
             '<text x="28" y="63">Two future slices shown; A and Q are latent. Full graph: 11 time points × 4 shared motion branches.</text>']
    def label(x, y, text, cls=''):
        parts.append(f'<text x="{x}" y="{y}" class="{cls}">{escape(str(text))}</text>')
    def box(x,y,w,h,fill,title,detail=''):
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{fill}" stroke="#64748b"/>')
        label(x+10,y+23,title)
        if detail: label(x+10,y+44,detail,'small')
    def line(x1,y1,x2,y2,cls='learned'):
        parts.append(f'<path d="M{x1},{y1} L{x2},{y2}" class="{cls}"/>')
    box(28,110,265,76,'#e2e8f0','Available cutoff history','Exact actor / station / flow records')
    box(28,211,265,76,'#e2e8f0','Frozen shared motion q','One state draw per actor / time / branch')
    box(28,324,265,86,'#e6fffa','History + motion encoder','Outputs fixed-basis coefficients only')
    line(160,186,160,324,'context'); line(200,287,200,324,'context')
    label(28,440,'No future A, Q or labels in this input.','small')
    label(28,462,'No free six-state unary or transition table.','small')
    for j,x in enumerate((440,835)):
        label(x,107,f'Future slice t+{j+1}')
        box(x,132,220,58,'#dbeafe','A: reduced availability','0 = nominal; 1 = availability < 1')
        box(x,400,220,58,'#dbeafe','Q: QoS interval','pass / loss-only / both fail')
        box(x-30,278,125,57,'#dcfce7','L: latency','above threshold')
        box(x+150,278,125,57,'#dcfce7','P: loss','above threshold')
        # Small square factors represent association functions, not entities.
        for name,px in (('AL',x+18),('AP',x+184)):
            on=active[name];color='#99f6e4' if on else '#e2e8f0';cls='learned' if on else 'off'
            parts.append(f'<rect x="{px}" y="222" width="24" height="24" fill="{color}" stroke="#64748b"/>')
            line(x+110,190,px+12,222,cls)
            line(px+12,246,x+32 if name=='AL' else x+212,278,cls)
            label(px-5,215,name,'small')
        for px,target in ((x+35,x+32),(x+180,x+212)):
            parts.append(f'<rect x="{px}" y="363" width="22" height="22" fill="#fed7aa" stroke="#26384c"/>')
            line(x+110,400,px+11,385,'hard');line(px+11,363,target,335,'hard')
        label(x+45,488,'Hard Q→L/P; fixed support','small')
    # Temporal factors have explicit two-variable scopes and centered bases.
    for name,y in (('AA',162),('QQ',430)):
        line(660,y,724,y);line(752,y,835,y)
        parts.append(f'<rect x="724" y="{y-14}" width="28" height="28" fill="#99f6e4" stroke="#087f8c"/>')
        label(720,y-22,name,'small')
    line(660,170,777,225);line(777,249,1025,278)
    parts.append('<rect x="765" y="225" width="24" height="24" fill="#99f6e4" stroke="#087f8c"/>')
    label(696,219,'A(t)–P(t+1)','small')
    line(535,308,695,342);line(719,342,835,180)
    parts.append('<rect x="695" y="330" width="24" height="24" fill="#99f6e4" stroke="#087f8c"/>')
    label(666,375,'L(t)–A(t+1)','small')
    line(293,365,378,365,'context')
    label(312,390,'coefficient','small');label(312,407,'conditioning','small')
    box(28,530,1164,100,'#ffffff','Fixed candidate structures: none / AL / AP / AL+AP',
        f'Validation selected {selected}; AL mean coefficient = {coefficients["AL"]["mean"]:.4f}; AP effective mean = {coefficients["AP"]["mean"]:.4f}')
    label(40,608,'Means summarize this exported window across branches and time; actual coefficients vary with context.','small')
    box(28,651,1164,90,'#ffffff','One joint partition per branch → latent marginalization → mean branch probabilities',
        'Outputs: L/P joint truth, first-flip distributions and A/Q posteriors. Train with joint censored-event and masked-A likelihood.')
    label(40,721,'Solid dark: fixed rules. Teal: learned potentials. Dashed grey: disabled candidate. Square: factor function.','small')
    label(28,782,f'Test event NLL: selected {result["test_metrics"]["selected"]["event_nll"]:.4f}; retrained no-cross {result["test_metrics"]["retrained_no_cross"]["event_nll"]:.4f}.')
    label(28,809,'Few independent scenario families; no future reduced-A labels in the fixed test subset. No causal claim.','small')
    label(28,835,'This view summarizes repeated scopes. Exact typed keys, source references, coefficients and knockouts accompany graph.json.','small')
    parts.append('</svg>')
    (output/'graph.svg').write_text('\n'.join(parts)+'\n')


if __name__ == '__main__':
    write_mixed_svg(Path(__file__).resolve().parents[3]/'Dataset/world_model/runs/rule_factor_graph')
