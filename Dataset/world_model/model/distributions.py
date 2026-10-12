"""Shared contextual trunk with typed, causally anchored probability laws."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, Gamma, Beta, VonMises
from .causal_input import ATOM_SEMANTICS, STATUS
from .query_head import QUERY_FEATURES


def law_group(f):
    if f.circular: return "circular"
    if f.lower is not None and f.upper is not None: return "bounded"
    if f.lower is not None: return "nonnegative"
    if f.upper is not None: raise ValueError("upper-only support is not declared in this inventory")
    return "normal"


class SharedDistributionHead(nn.Module):
    def __init__(self, hidden=2048, width=128, max_classes=5):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(hidden*3+QUERY_FEATURES, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.typed = nn.ModuleDict({k: nn.Linear(width, n) for k, n in
                                   {"normal":3,"circular":2,"bounded":5,"nonnegative":3}.items()})
        self.classes = nn.Linear(width, max_classes)
        self.status = nn.Linear(width, len(STATUS))
        self.recording = nn.Linear(width, 1)
        for output in (*self.typed.values(), self.classes, self.status, self.recording):
            nn.init.zeros_(output.weight); nn.init.zeros_(output.bias)
        self.max_classes = max_classes

    def forward(self, query, entity_context, field_context, global_context):
        n = query["features"].shape[0]
        context = torch.cat((query["features"], entity_context.float(), field_context.float(),
                             global_context.float().expand(n, -1)), -1)
        hidden = self.trunk(context)
        raw = {k: layer(hidden) for k, layer in self.typed.items()}
        parameters = torch.stack([F.pad(raw[law_group(f)][i], (0, 6-raw[law_group(f)].shape[1]))
                                  if f.logical_dtype == "real" else hidden.new_zeros(6)
                                  for i, f in enumerate(query["fields"])])
        class_prior = F.one_hot(query["category_prior"], self.max_classes).float()*4.
        status_prior = hidden.new_zeros((n, len(STATUS))); status_prior[:,0] = 4.
        return {"parameters": parameters, "classes": self.classes(hidden)+class_prior,
                "status": (self.status(hidden)+status_prior).masked_fill(~query["allowed_status"], -torch.inf),
                "recorded": self.recording(hidden).squeeze(-1)+(query["features"][:,2]*2.-1.)*4.}


# Softplus offset that maps a zero-initialised raw scale to exactly one unit.
UNIT_SCALE_OFFSET = math.log(math.e-1.)
# Floor of a Gaussian standard deviation, in fitted units.  Most TRAIN cells of
# the dynamic pose fields equal their anchor exactly (stationary or constant-
# velocity actors), where the Gaussian NLL has no finite optimum; the floor
# fixes that optimum and bounds the gradient scale.
GAUSSIAN_SCALE_FLOOR = 0.05


def law(parameters, f, anchor, forecast_std, positive_prior):
    """Typed law of one queried cell, or of a batch of cells sharing ``f``'s support.

    ``parameters`` is ``[..., 6]``; the other tensors carry the leading shape.
    ``forecast_std`` is the cell's fitted physical unit (TRAIN residual rate
    times elapsed seconds).  Unbounded real fields are anchor-inflated
    Gaussians: a point mass at the anchor (no change from the extrapolated
    state) plus a Gaussian whose mean offset and standard deviation the head
    predicts directly in that unit.  ``mass`` is (log P(atom), log P(off atom)).
    """
    # Float64 avoids cancellation in Beta/Gamma log normalizers for small
    # genuine temporal ratio changes; gradients return to FP32 heads.
    p, anchor, scale = parameters.double(), anchor.double(), forecast_std.double()
    if f.circular:
        location = torch.deg2rad(anchor+p[..., 0]*scale)
        concentration = torch.rad2deg(scale.new_tensor(1.)).square()/scale.square()*torch.exp(p[..., 1]*.1)
        return "von_mises", VonMises(location, concentration), None
    if f.lower is not None and f.upper is not None:
        mean = (anchor-f.lower)/(f.upper-f.lower)
        epsilon = torch.finfo(mean.dtype).eps
        interior_mean = mean.clamp(epsilon, 1.-epsilon)
        interior_mean = torch.sigmoid(torch.logit(interior_mean)+p[..., 0])
        scaled_std = scale/(f.upper-f.lower)
        concentration = interior_mean*(1.-interior_mean)/scaled_std.square()*torch.exp(p[..., 1]*.1)
        alpha, beta = interior_mean*concentration, (1.-interior_mean)*concentration
        prior = torch.stack((torch.where(mean == 0, mean.new_tensor(8.), mean.new_tensor(-16.)),
                             torch.where(mean == 1, mean.new_tensor(8.), mean.new_tensor(-16.)), torch.zeros_like(mean)), -1)
        return "endpoint_inflated_beta", Beta(alpha, beta), torch.log_softmax(prior+p[..., 2:5], -1)
    if f.lower is not None:
        interior_mean = torch.where(anchor > f.lower, anchor-f.lower, positive_prior.double())*torch.exp(p[..., 0]*.1)
        shape = interior_mean.square()/scale.square()*torch.exp(p[..., 1]*.1)+.2
        rate = shape/interior_mean
        zero_logit = p[..., 2]+torch.where(anchor == f.lower, anchor.new_tensor(8.), anchor.new_tensor(-8.))
        return "boundary_inflated_gamma", Gamma(shape, rate), torch.stack((F.logsigmoid(zero_logit), F.logsigmoid(-zero_logit)), -1)
    gaussian = Normal(anchor+p[..., 0]*scale, scale*(F.softplus(p[..., 1]+UNIT_SCALE_OFFSET)+GAUSSIAN_SCALE_FLOOR))
    return "anchor_inflated_normal", gaussian, torch.stack((F.logsigmoid(p[..., 2]), F.logsigmoid(-p[..., 2])), -1)


def band_mass(distribution, anchor, band):
    """Gaussian mass inside the anchor band [anchor-band, anchor+band], the log mass of its complement, and the bounds."""
    loc, scale = distribution.loc, distribution.scale
    lower, upper = (anchor.double()-band-loc)/scale, (anchor.double()+band-loc)/scale
    # Complement = lower tail + upper tail; finite even when the band holds almost all mass.
    log_complement = torch.logaddexp(torch.special.log_ndtr(lower), torch.special.log_ndtr(-upper))
    return -torch.expm1(log_complement), log_complement, lower, upper


def cell_nll(name, distribution, mass, value, f, anchor=None, atom=None, band=None):
    """Negative log likelihood of present real values under their typed law (physical units).

    Boundary atoms take their mass; the continuous part is evaluated only where
    the value is interior (``torch.where`` keeps its gradient off the atoms).
    Unbounded fields observe the anchor band as one censored outcome with
    probability P(atom); off the band the Gaussian is normalized over the band's
    complement.  ``atom`` is the exact label from ``causal_input.anchor_atom``.
    """
    if name == "von_mises":
        return -distribution.log_prob(torch.deg2rad(value))-math.log(math.pi/180.)
    if name == "anchor_inflated_normal":
        _, log_complement, _, _ = band_mass(distribution, anchor, band)
        off_anchor = -mass[..., 1]-distribution.log_prob(value)+log_complement
        return torch.where(atom, -mass[..., 0], off_anchor)
    if name == "boundary_inflated_gamma":
        at_lower = value == f.lower
        interior = torch.where(at_lower, value.new_ones(()), value-f.lower)
        return torch.where(at_lower, -mass[..., 0], -mass[..., 1]-distribution.log_prob(interior))
    if name == "endpoint_inflated_beta":
        at_lower, at_upper = value == f.lower, value == f.upper
        interior = torch.where(at_lower | at_upper, value.new_full((), .5), (value-f.lower)/(f.upper-f.lower))
        continuous = -mass[..., 2]-distribution.log_prob(interior)+math.log(f.upper-f.lower)
        return torch.where(at_lower, -mass[..., 0], torch.where(at_upper, -mass[..., 1], continuous))
    raise ValueError(f"undeclared law {name}")


def mixture_crps(anchor, p_atom, loc, scale, value):
    """CRPS of p*delta_anchor + (1-p)*Normal(loc, scale) at ``value`` (physical units).

    Closed form from E|X-y| - E|X-X'|/2 with A(d, s) = E|d + s Z|.  The
    band-conditioned Gaussian differs from the plain one by the band's mass
    (< 1e-7 in aggregate CRPS on VALID); atom outcomes are scored at the anchor.
    """
    q = 1-p_atom
    def expected_abs(d, s):
        z = d/s
        return 2*s*torch.exp(-z*z/2)/math.sqrt(2*math.pi)+d*(2*torch.special.ndtr(z)-1)
    return (p_atom*(anchor-value).abs()+q*expected_abs(loc-value, scale)
            - p_atom*q*expected_abs(loc-anchor, scale)-q*q*scale/math.sqrt(math.pi))


def typed_loss(output, query, target, crps_weight=0.):
    status_rows = target["status_mask"]
    recording = F.binary_cross_entropy_with_logits(output["recorded"], target["recorded"])
    status = F.cross_entropy(output["status"][status_rows], target["status"][status_rows]) if bool(status_rows.any()) else output["parameters"].sum()*0.
    # Cells sharing a law support are evaluated together; the losses stay the
    # mean of per-cell NLL/CE values.
    present = (status_rows & (target["status"] == 0)).tolist()
    real_groups, class_groups = {}, {}
    for i, f in enumerate(query["fields"]):
        if not present[i]:
            continue
        if f.logical_dtype == "real":
            real_groups.setdefault((f.circular, f.lower, f.upper), (f, []))[1].append(i)
        else:
            class_groups.setdefault(len(f.enum_values), []).append(i)
    numeric, categorical, crps = [], [], []
    for f, rows in real_groups.values():
        rows = torch.tensor(rows, device=output["parameters"].device)
        name, distribution, mass = law(output["parameters"][rows], f, query["anchor"][rows], query["forecast_std"][rows],
                                       query["positive_prior"][rows])
        numeric.append(cell_nll(name, distribution, mass, target["numeric"][rows].double(), f, query["anchor"][rows],
                                target["atom"][rows], query["atom_band"][rows]))
        if crps_weight and name == "anchor_inflated_normal":
            anchor = query["anchor"][rows].double()
            value = torch.where(target["atom"][rows], anchor, target["numeric"][rows].double())
            # Scale-free: CRPS in units of the cell's fitted physical unit.
            crps.append(mixture_crps(anchor, mass[..., 0].exp(), distribution.loc, distribution.scale, value)
                        / query["forecast_std"][rows].double())
    for count, rows in class_groups.items():
        rows = torch.tensor(rows, device=output["classes"].device)
        categorical.append(F.cross_entropy(output["classes"][rows, :count], target["categorical"][rows], reduction="none"))
    value_loss = torch.cat(numeric).mean() if numeric else output["parameters"].sum()*0.
    category_loss = torch.cat(categorical).mean() if categorical else output["classes"].sum()*0.
    result = {"typed":recording+status+value_loss+category_loss,"recording_bce":recording,
              "cell_kind_ce":status,"numeric_nll":value_loss,"category_ce":category_loss}
    if crps_weight:
        crps_loss = torch.cat(crps).mean() if crps else output["parameters"].sum()*0.
        result["scaled_crps"] = crps_loss
        result["typed"] = result["typed"]+crps_weight*crps_loss
    return result


def anchor_mixture_summary(anchor, p, off, distribution, band, levels=(.05, .25, .5, .75, .95)):
    """Moments and quantiles of p*[anchor band] + off*Gaussian off the band (p + off = 1).

    ``p`` and ``off`` are passed separately so that neither is recovered from
    the other by subtraction.  The band (half-width <= 1.5e-6) is treated as
    the point ``anchor`` for moments and quantiles.  Returns the
    full-distribution mean, variance and quantiles, plus quantiles of the
    off-anchor component alone.
    """
    anchor, p, off = anchor.double(), p.double(), off.double()
    mu, sigma = distribution.loc, distribution.scale
    inside, log_complement, lower, upper = band_mass(distribution, anchor, band)
    complement = log_complement.exp()
    if bool((complement < 1e-12).any()):
        raise ValueError("off-anchor Gaussian lies inside the anchor band; its conditional summary is undefined")
    pdf = lambda z: torch.exp(-.5*z*z)/math.sqrt(2*math.pi)
    shift = (pdf(upper)-pdf(lower))/complement
    off_mean = mu+sigma*shift
    off_variance = sigma**2*(1+(upper*pdf(upper)-lower*pdf(lower))/complement-shift**2)
    mean = p*anchor+off*off_mean
    variance = p*(anchor-mean)**2+off*(off_variance+(off_mean-mean)**2)
    below = off*torch.special.ndtr(lower)/complement
    def full(q):
        left = mu+sigma*torch.special.ndtri((q*complement/off).clamp(1e-300, 1-1e-16))
        right = mu+sigma*torch.special.ndtri((inside+(q-p)*complement/off).clamp(1e-300, 1-1e-16))
        return torch.where(q < below, left, torch.where(q <= below+p, anchor, right))
    def off_quantile(q):
        level = q*complement
        return mu+sigma*torch.special.ndtri(torch.where(level < torch.special.ndtr(lower), level, level+inside))
    return {"mean": mean, "variance": variance, "quantiles": {q: full(anchor.new_tensor(q)) for q in levels},
            "off_anchor_quantiles": {q: off_quantile(anchor.new_tensor(q)) for q in levels}}


def predictions(output, query):
    rows, rollout = [], {}
    for i, (a, f) in enumerate(zip(query["addresses"], query["fields"])):
        kind = STATUS[int(output["status"][i].argmax())]
        row = {"episode":a.episode_id,"entity":a.entity_id,"field":a.field_family,"component":a.component,
               "target_tick":a.target_tick,"record_probability":float(output["recorded"][i].sigmoid()),
               "cell_kind_probabilities":{k:float(v) for k,v in zip(STATUS,output["status"][i].softmax(-1))},"predicted_cell_kind":kind}
        if f.logical_dtype == "real":
            name, d, mass = law(output["parameters"][i],f,query["anchor"][i],query["forecast_std"][i],query["positive_prior"][i])
            if name == "anchor_inflated_normal":
                p=mass.exp(); anchor=float(query["anchor"][i])
                value=anchor if bool(p[0] >= .5) else float(d.mean)
                summary=anchor_mixture_summary(query["anchor"][i],p[0],p[1],d,query["atom_band"][i])
                full,off=summary["quantiles"],summary["off_anchor_quantiles"]
                params={"anchor":anchor,"atom_band":float(query["atom_band"][i]),"atom_semantics":ATOM_SEMANTICS,
                        "anchor_probability":float(p[0]),"anchor_log_probability":float(mass[0]),
                        "off_anchor_log_probability":float(mass[1]),"location":float(d.mean),"scale":float(d.scale),
                        "mixture_mean":float(summary["mean"]),"mixture_variance":float(summary["variance"]),
                        "mixture_median":float(full[.5]),
                        "interval_50":[float(full[.25]),float(full[.75])],"interval_90":[float(full[.05]),float(full[.95])],
                        "off_anchor_interval_50":[float(off[.25]),float(off[.75])],
                        "off_anchor_interval_90":[float(off[.05]),float(off[.95])]}
            elif name == "von_mises":
                value=float(torch.rad2deg(torch.atan2(d.loc.sin(),d.loc.cos())))
                params={"location_deg":value,"concentration":float(d.concentration)}
            elif name == "boundary_inflated_gamma":
                p=mass.exp(); value=f.lower if bool(p[0] >= .5) else float(f.lower+d.mean)
                params={"lower":f.lower,"boundary_mass":float(p[0]),"concentration":float(d.concentration),"rate":float(d.rate)}
            else:
                p=mass.exp()
                value=f.lower if bool(p[0] >= .5) else f.upper if bool(p[1] >= .5) else float(f.lower+(f.upper-f.lower)*d.mean)
                params={"bounds":[f.lower,f.upper],"endpoint_masses":p[:2].tolist(),"interior_mass":float(p[2]),"alpha":float(d.concentration1),"beta":float(d.concentration0)}
            row.update({"law":name,"parameters":params,"conditional_point":value,
                        "point_rule":"hard selection for rollout feedback: anchor or boundary atom when its mass>=0.5, otherwise continuous-component mean; circular mean for angles (not the mixture mean or median)"})
        else:
            probs=output["classes"][i,:len(f.enum_values)].softmax(-1); value=f.enum_values[int(probs.argmax())]
            row.update({"law":"categorical","values":list(f.enum_values),"probabilities":probs.tolist(),"conditional_point":value})
        if a.entity_id not in rollout: rollout[a.entity_id]={}
        rollout[a.entity_id][f.family,f.component]={"value":value if kind == "present" else None,
                                                  "kind":kind,"record_probability":row["record_probability"]}
        rows.append(row)
    return rows, rollout
