import networkx as nx


def solve_pricing_problem(ship, duals_fulfill, duals_sp, duals_bs, total_steps, allowed_mode='all'):
    G = nx.DiGraph()
    source, sink = 'Src', 'Snk'
    pi_val = duals_fulfill

    # ... (后面的代码完全保持不变) ...
    G.add_edge(source, ship.arrival_time, weight=0, mode='start')

    for t in range(ship.arrival_time, ship.deadline):
        if t + 1 <= ship.deadline:
            G.add_edge(t, t + 1, weight=0, mode='wait')

        candidates = []

        # A. 岸电
        if allowed_mode in ['all', 'shore']:
            duration_sp = max(ship.t_cargo, ship.t_shore_power)
            end_t_sp = t + duration_sp
            if end_t_sp <= ship.deadline and end_t_sp <= total_steps:
                limit_sp = min(end_t_sp, len(duals_sp))
                pen_sp = sum(duals_sp[k] for k in range(t, limit_sp)) if t < limit_sp else 0
                w_sp = ship.cost_shore - pen_sp
                candidates.append({'weight': w_sp, 'mode': 'shore', 'dur': duration_sp, 'end': end_t_sp})

        # B. 换电
        if allowed_mode in ['all', 'battery']:
            duration_bs = max(ship.t_cargo, ship.t_battery_swap)
            end_t_bs = t + duration_bs
            if end_t_bs <= ship.deadline and end_t_bs <= total_steps:
                pen_bs = duals_bs[t] if t < len(duals_bs) else 0
                w_bs = ship.cost_battery - pen_bs
                candidates.append({'weight': w_bs, 'mode': 'battery', 'dur': duration_bs, 'end': end_t_bs})

        # 择优
        if candidates:
            best_cand = min(candidates, key=lambda x: x['weight'])
            G.add_edge(t, sink, weight=best_cand['weight'], mode=best_cand['mode'], start_t=t, dur=best_cand['dur'])

    try:
        path = nx.bellman_ford_path(G, source, sink, weight='weight')
        path_len = nx.path_weight(G, path, weight='weight')
        rc = path_len - pi_val

        if rc < -1e-6:
            start_node = path[-2]
            edge_data = G.get_edge_data(start_node, sink)
            return {
                'rc': rc,
                'mode': edge_data['mode'],
                'start': edge_data['start_t'],
                'duration': edge_data['dur'],
                'cost': ship.cost_shore if edge_data['mode'] == 'shore' else ship.cost_battery
            }
        return {"rc": rc}
    except nx.NetworkXNoPath:
        return {"rc": 0.0}
