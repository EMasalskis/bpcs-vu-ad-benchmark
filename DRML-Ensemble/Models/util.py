try:
    import torch
    from torch.distributions import Bernoulli
except ImportError:
    pass


class MyBaseTransform:
    def __call__(self, g, c_etype):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__ + '()'


class MyDropEdge(MyBaseTransform):
    def __init__(self, p=0.5, device="cuda:0"):
        self.p = p
        self.dist = Bernoulli(p)
        self.device = device

    def __call__(self, g, c_etypes):
        # Fast path
        if self.p == 0:
            return g

        for c_etype in c_etypes:
            samples = self.dist.sample(torch.Size([g.num_edges(c_etype)]))
            eids_to_remove = g.edges(form='eid', etype=c_etype)[samples.bool().to(g.device)]
            # g.remove_edges(eids_to_remove, etype=c_etype)
            mask = torch.ones((g.num_edges(c_etype))).to(self.device)
            mask[eids_to_remove] = 0
            g.edges[c_etype].data["mask"] = mask
        return g

