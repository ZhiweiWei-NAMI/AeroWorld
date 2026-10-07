"""Shared contextual trunk with typed, causally anchored probability laws."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, Gamma, Beta, VonMises
from .causal_input import STATUS


def law_group(f):
    if f.circular: return "circular"
    if f.lower is not None and f.upper is not None: return "bounded"
    if f.lower is not None: return "nonnegative"
    if f.upper is not None: raise ValueError("upper-only support is not declared in this inventory")
    return "normal"


class SharedDistributionHead(nn.Module):
    def __init__(self, hidden=2048, width=128, max_classes=5):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(hidden*3+12, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.typed = nn.ModuleDict({k: nn.Linear(width, n) for k, n in
                                   {"normal":2,"circular":2,"bounded":5,"nonnegative":3}.items()})
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


def law(parameters, f, anchor, forecast_std, positive_prior):
    # Float64 avoids cancellation in Beta/Gamma log normalizers for small
    # genuine temporal ratio changes; gradients return to FP32 heads.
    p, anchor, scale = parameters.double(), anchor.double(), forecast_std.double()
    if f.circular:
        location = torch.deg2rad(anchor+p[0]*scale)
        concentration = torch.rad2deg(scale.new_tensor(1.)).square()/scale.square()*torch.exp(p[1]*.1)
        return "von_mises", VonMises(location, concentration), None
    if f.lower is not None and f.upper is not None:
        mean = (anchor-f.lower)/(f.upper-f.lower)
        epsilon = torch.finfo(mean.dtype).eps
        interior_mean = mean.clamp(epsilon, 1.-epsilon)
        interior_mean = torch.sigmoid(torch.logit(interior_mean)+p[0])
        scaled_std = scale/(f.upper-f.lower)
        concentration = interior_mean*(1.-interior_mean)/scaled_std.square()*torch.exp(p[1]*.1)
        alpha, beta = interior_mean*concentration, (1.-interior_mean)*concentration
        prior = torch.stack((torch.where(mean == 0, mean.new_tensor(8.), mean.new_tensor(-16.)),
                             torch.where(mean == 1, mean.new_tensor(8.), mean.new_tensor(-16.)), mean.new_tensor(0.)))
        return "endpoint_inflated_beta", Beta(alpha, beta), torch.log_softmax(prior+p[2:5], 0)
    if f.lower is not None:
        interior_mean = torch.where(anchor > f.lower, anchor-f.lower, positive_prior.double())*torch.exp(p[0]*.1)
        shape = interior_mean.square()/scale.square()*torch.exp(p[1]*.1)+.2
        rate = shape/interior_mean
        zero_logit = p[2]+torch.where(anchor == f.lower, anchor.new_tensor(8.), anchor.new_tensor(-8.))
        return "boundary_inflated_gamma", Gamma(shape, rate), torch.stack((F.logsigmoid(zero_logit), F.logsigmoid(-zero_logit)))
    return "normal", Normal(anchor+p[0]*scale, scale*torch.exp(p[1]*.1)), None


def typed_loss(output, query, target):
    status_rows = target["status_mask"]
    recording = F.binary_cross_entropy_with_logits(output["recorded"], target["recorded"])
    status = F.cross_entropy(output["status"][status_rows], target["status"][status_rows]) if bool(status_rows.any()) else output["parameters"].sum()*0.
    numeric, categorical = [], []
    for i, f in enumerate(query["fields"]):
        if not bool(status_rows[i]) or int(target["status"][i]) != 0: continue
        if f.logical_dtype != "real":
            categorical.append(F.cross_entropy(output["classes"][i,:len(f.enum_values)].unsqueeze(0), target["categorical"][i:i+1]))
            continue
        name, distribution, mass = law(output["parameters"][i], f, query["anchor"][i], query["forecast_std"][i], query["positive_prior"][i])
        value = target["numeric"][i].double()
        if name == "von_mises": nll = -distribution.log_prob(torch.deg2rad(value))-math.log(math.pi/180.)
        elif name == "boundary_inflated_gamma": nll = -mass[0] if bool(value == f.lower) else -mass[1]-distribution.log_prob(value-f.lower)
        elif name == "endpoint_inflated_beta":
            if bool(value == f.lower): nll = -mass[0]
            elif bool(value == f.upper): nll = -mass[1]
            else: nll = -mass[2]-distribution.log_prob((value-f.lower)/(f.upper-f.lower))+math.log(f.upper-f.lower)
        else: nll = -distribution.log_prob(value)
        numeric.append(nll)
    value_loss = torch.stack(numeric).mean() if numeric else output["parameters"].sum()*0.
    category_loss = torch.stack(categorical).mean() if categorical else output["classes"].sum()*0.
    return {"typed":recording+status+value_loss+category_loss,"recording_bce":recording,
            "cell_kind_ce":status,"numeric_nll":value_loss,"category_ce":category_loss}


def predictions(output, query):
    rows, rollout = [], {}
    for i, (a, f) in enumerate(zip(query["addresses"], query["fields"])):
        kind = STATUS[int(output["status"][i].argmax())]
        row = {"episode":a.episode_id,"entity":a.entity_id,"field":a.field_family,"component":a.component,
               "target_tick":a.target_tick,"record_probability":float(output["recorded"][i].sigmoid()),
               "cell_kind_probabilities":{k:float(v) for k,v in zip(STATUS,output["status"][i].softmax(-1))},"predicted_cell_kind":kind}
        if f.logical_dtype == "real":
            name, d, mass = law(output["parameters"][i],f,query["anchor"][i],query["forecast_std"][i],query["positive_prior"][i])
            if name == "normal": value=float(d.mean); params={"location":value,"scale":float(d.scale)}
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
                        "point_rule":"boundary atom when mass>=0.5; otherwise continuous-component mean; circular mean for angles"})
        else:
            probs=output["classes"][i,:len(f.enum_values)].softmax(-1); value=f.enum_values[int(probs.argmax())]
            row.update({"law":"categorical","values":list(f.enum_values),"probabilities":probs.tolist(),"conditional_point":value})
        if a.entity_id not in rollout: rollout[a.entity_id]={}
        rollout[a.entity_id][f.family,f.component]={"value":value,"kind":kind,"record_probability":row["record_probability"]}
        rows.append(row)
    return rows, rollout
