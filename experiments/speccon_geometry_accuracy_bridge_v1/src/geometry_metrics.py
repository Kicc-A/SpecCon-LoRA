"""Preregistered map-level geometry scores."""
import torch


def recurrence(ua, ub, rank):
    assert ua.ndim == ub.ndim == 2 and ua.shape[0] == ub.shape[0]
    assert 1 <= rank <= min(ua.shape[1], ub.shape[1])
    a, b = ua[:, :rank], ub[:, :rank]
    eye = torch.eye(rank, dtype=a.dtype, device=a.device)
    assert torch.allclose(a.T @ a, eye, atol=2e-3, rtol=0)
    assert torch.allclose(b.T @ b, eye, atol=2e-3, rtol=0)
    value = float((a.T @ b).square().sum() / rank)
    assert -1e-5 <= value <= 1+1e-5
    return min(1.0, max(0.0, value))


def budget(rank_map, shapes):
    assert rank_map.keys() == shapes.keys()
    return sum(int(rank_map[n]) * sum(shapes[n]) for n in rank_map)


def score_map(rank_map, a_bases, b_bases, b_grads):
    assert rank_map.keys() == a_bases.keys() == b_bases.keys() == b_grads.keys()
    shapes={n:tuple(g.shape) for n,g in b_grads.items()}
    total=budget(rank_map,shapes)
    geom_num=grad_num=grad_den=0.0
    details=[]
    for name, rank in rank_map.items():
        ua, ub, gb=a_bases[name], b_bases[name], b_grads[name]
        assert gb.ndim==2 and gb.shape[0]==ua.shape[0]==ub.shape[0]
        r=recurrence(ua,ub,rank)
        cost=sum(gb.shape)
        baseline=rank/gb.shape[0]
        covered=float((ua[:,:rank].T @ gb).square().sum())
        energy=float(gb.square().sum())
        geom_num+=cost*rank*(r-baseline)
        grad_num+=covered
        grad_den+=energy
        details.append({'module':name,'rank':rank,'cost':cost,'recurrence':r,
                        'isotropic_baseline':baseline,'covered_energy':covered,
                        'total_energy':energy})
    assert total>0 and grad_den>0
    q_geom=geom_num/total
    q_grad=grad_num/grad_den
    assert -1e-5<=q_grad<=1+1e-4
    return {'q_geom':q_geom,'q_grad':min(1.0,max(0.0,q_grad)),
            'adapter_budget':total,'details':details}
