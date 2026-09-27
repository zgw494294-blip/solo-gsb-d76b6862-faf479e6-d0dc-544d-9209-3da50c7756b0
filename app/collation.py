"""异文对读台：全局最小编辑距离对齐与异文汇总。

对齐规则
--------
- 匹配成本 0；替换、删除、插入成本均为 1。
- 并列（同为最小成本）路径按操作序列的字典序取舍，
  操作的字典序为：匹配 < 替换 < 删除 < 插入。
- 空白与标点一律照常参与比较，不做任何规范化。

坐标体系
--------
- 字符坐标：底本第 1 字 … 第 m 字（1 起计）。
- 间隙坐标：间隙 0（文首）… 间隙 m（文末），间隙 g 位于第 g 字之后、
  第 g+1 字之前，首尾间隙一并计入。

共享快照
--------
各接口的对齐结果（最小代价表、字典序首条路径及其坐标读法）经进程内
共享快照缓存取用：以原样 (底本, 文本) 为键，最多 8 份，LRU 淘汰，
同键并发只计算一次；命中、淘汰与并发下结果与现算一致。
"""

from .aligncache import AlignmentSnapshotCache

OP_MATCH = "match"
OP_SUBSTITUTE = "substitute"
OP_DELETE = "delete"
OP_INSERT = "insert"

KIND_LABELS = {
    OP_MATCH: "相同",
    OP_SUBSTITUTE: "替换",
    OP_DELETE: "缺字",
    OP_INSERT: "插入",
}


class EditionError(ValueError):
    """校定选择无效（重复坐标、无效坐标或读法不属于该坐标）。"""


def _cost_table(base: str, witness: str):
    """最小编辑代价表：cost[i][j] = 对齐 base[i:] 与 witness[j:] 的最小成本。"""
    m, n = len(base), len(witness)
    cost = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m - 1, -1, -1):
        cost[i][n] = cost[i + 1][n] + 1
    for j in range(n - 1, -1, -1):
        cost[m][j] = cost[m][j + 1] + 1
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            diag = (0 if base[i] == witness[j] else 1) + cost[i + 1][j + 1]
            dele = 1 + cost[i + 1][j]
            ins = 1 + cost[i][j + 1]
            cost[i][j] = min(diag, dele, ins)
    return cost


def _viable_ops(base: str, witness: str, cost: list, i: int, j: int):
    """(i, j) 处能走通最优解的操作，按字典序（匹配＜替换＜删除＜插入）排列。"""
    m, n = len(base), len(witness)
    ops = []
    if i < m and j < n:
        if base[i] == witness[j] and cost[i][j] == cost[i + 1][j + 1]:
            ops.append(OP_MATCH)
        if cost[i][j] == 1 + cost[i + 1][j + 1]:
            ops.append(OP_SUBSTITUTE)
    if i < m and cost[i][j] == 1 + cost[i + 1][j]:
        ops.append(OP_DELETE)
    if j < n and cost[i][j] == 1 + cost[i][j + 1]:
        ops.append(OP_INSERT)
    return ops


def _step_event(op: str, i: int, j: int):
    """在 (i, j) 处执行操作，返回 (事件, 新 i, 新 j)。"""
    if op in (OP_MATCH, OP_SUBSTITUTE):
        return (op, i, j), i + 1, j + 1
    if op == OP_DELETE:
        return (op, i, None), i + 1, j
    return (op, None, j), i, j + 1


def _greedy_events(base: str, witness: str, cost: list):
    """自 (0, 0) 起贪心走最小成本路径：每步在“可走通最优解”的操作里
    选字典序最小者（匹配 < 替换 < 删除 < 插入），即得操作序列字典序
    最小的最优路径。"""
    m, n = len(base), len(witness)
    events = []
    i = j = 0
    while i < m or j < n:
        op = _viable_ops(base, witness, cost, i, j)[0]
        event, i, j = _step_event(op, i, j)
        events.append(event)
    return events


def align(base: str, witness: str):
    """把抄本全局对齐到底本，返回操作事件列表（自左向右）。

    每个事件为三元组 (op, i, j)：
      ("match",      i, j)  底本 base[i] 与抄本 witness[j] 相同
      ("substitute", i, j)  底本 base[i] 被抄本 witness[j] 替换
      ("delete",     i, None) 底本 base[i] 在抄本中缺字
      ("insert",     None, j) 抄本 witness[j] 为插入字
    i、j 均为 0 起索引。
    """
    return list(_snapshot(base, witness).events)


def _path_count(base: str, witness: str, cost: list) -> int:
    """精确统计最小代价操作路径数：不同操作序列算不同路径。"""
    m, n = len(base), len(witness)
    # count[i][j] = 自 (i, j) 起走完对齐的最优路径数
    count = [[0] * (n + 1) for _ in range(m + 1)]
    count[m][n] = 1
    for i in range(m, -1, -1):
        for j in range(n, -1, -1):
            if i == m and j == n:
                continue
            total = 0
            for op in _viable_ops(base, witness, cost, i, j):
                if op in (OP_MATCH, OP_SUBSTITUTE):
                    total += count[i + 1][j + 1]
                elif op == OP_DELETE:
                    total += count[i + 1][j]
                else:  # OP_INSERT
                    total += count[i][j + 1]
            count[i][j] = total
    return count[0][0]


def _second_events(base: str, witness: str, cost: list, first_events: list):
    """构造操作序列字典序严格次小的最优路径。

    首条路径每步都取字典序最小的可行操作；其他任何最优路径与首条路径
    的首次分叉处，必改取字典序更大的可行操作。分叉越晚、其后全取最小，
    所得序列字典序越小。故重走首条路径，在最晚一个存在次小可行操作的
    位置改取次小操作，其后逐步取最小，即得严格次小的最优路径。
    路径唯一时返回 None。
    """
    m, n = len(base), len(witness)
    states = []  # 首条路径每步的出发位置与可行操作
    i = j = 0
    for op, _a, _b in first_events:
        states.append((i, j, _viable_ops(base, witness, cost, i, j)))
        _event, i, j = _step_event(op, i, j)
    for t in range(len(states) - 1, -1, -1):
        si, sj, viable = states[t]
        if len(viable) < 2:
            continue
        events = list(first_events[:t])
        event, i, j = _step_event(viable[1], si, sj)
        events.append(event)
        while i < m or j < n:
            op = _viable_ops(base, witness, cost, i, j)[0]
            event, i, j = _step_event(op, i, j)
            events.append(event)
        return events
    return None


def _readings_from_events(base: str, witness: str, events: list):
    """按底本坐标整理一份抄本的读法。

    返回 (char_readings, gap_readings)：
      char_readings[i]：底本第 i+1 字处，该抄本的读法；
      gap_readings[g]：间隙 g 处，该抄本的插入情况。
    每条读法为 {"kind", "reading", "span"}，span 为抄本内的
    0 起半开区间 [start, end)；空区间表示抄本中的对应间隙位置。
    """
    m = len(base)
    char_readings = [None] * m
    gap_readings = [None] * (m + 1)

    i = j = 0
    opened_gap = -1  # 已记录到哪个间隙
    for op, a, b in events:
        if opened_gap < i:
            gap_readings[i] = {"kind": OP_MATCH, "reading": "", "span": [j, j]}
            opened_gap = i
        if op == OP_MATCH:
            char_readings[i] = {
                "kind": OP_MATCH,
                "reading": base[i],
                "span": [j, j + 1],
            }
            i += 1
            j += 1
        elif op == OP_SUBSTITUTE:
            char_readings[i] = {
                "kind": OP_SUBSTITUTE,
                "reading": witness[b],
                "span": [b, b + 1],
            }
            i += 1
            j += 1
        elif op == OP_DELETE:
            char_readings[i] = {"kind": OP_DELETE, "reading": "", "span": [j, j]}
            i += 1
        else:  # OP_INSERT
            gap = gap_readings[i]
            if gap["kind"] == OP_MATCH:
                gap["kind"] = OP_INSERT
                gap["reading"] = witness[b]
                gap["span"] = [b, b + 1]
            else:
                gap["reading"] += witness[b]
                gap["span"][1] = b + 1
            j += 1
    for g in range(opened_gap + 1, m + 1):
        gap_readings[g] = {"kind": OP_MATCH, "reading": "", "span": [j, j]}
    return char_readings, gap_readings


def _compute_snapshot(key):
    """算出 (base, text) 的对齐快照正本：最小代价表、字典序首条路径
    及其坐标读法。代价表与路径事件转为元组（不可变）后由缓存共享；
    坐标读法正本由缓存层深拷贝后交付调用方。"""
    base, text = key
    cost = _cost_table(base, text)
    events = _greedy_events(base, text, cost)
    readings = _readings_from_events(base, text, events)
    return (
        tuple(tuple(row) for row in cost),
        tuple(events),
        readings,
    )


# 进程内共享对齐快照：以原样 (底本, 文本) 为键（抄本名不进键），
# 最多保留 8 份已完成快照，按最近最少使用淘汰
_SNAPSHOTS = AlignmentSnapshotCache(capacity=8, compute=_compute_snapshot)


def _snapshot(base: str, text: str):
    """取 (base, text) 的共享对齐快照视图；未命中或已淘汰则现算。"""
    return _SNAPSHOTS.get((base, text))


def _witness_readings(base: str, witness: str):
    """现行首条路径（align 的字典序取舍）下的抄本读法。"""
    return _snapshot(base, witness).readings


def _gap_label(g: int, m: int) -> str:
    if g == 0:
        return "文首间隙"
    if g == m:
        return "文末间隙"
    return f"第{g}字后间隙"


def _iter_coordinates(base: str):
    """按「间隙0、第1字、间隙1、……、第m字、间隙m」的顺序遍历底本坐标。

    每项为 (type_, index, coordinate, base_reading, base_span)：
      间隙坐标 type_="gap"，底本读法为空串，base_span 为零长区间 [g, g)；
      字符坐标 type_="char"，底本读法为底本该字，base_span 为 [i, i+1)。
    index 即对外坐标序号：间隙为 0…m，字符为 1…m。
    """
    m = len(base)
    for g in range(m + 1):
        yield (
            "gap",
            g,
            {"type": "gap", "index": g, "label": _gap_label(g, m)},
            "",
            [g, g],
        )
        if g < m:
            yield (
                "char",
                g + 1,
                {"type": "char", "index": g + 1, "label": f"第{g + 1}字"},
                base[g],
                [g, g + 1],
            )


def collate(base: str, witnesses: list) -> dict:
    """汇总各抄本与底本的异文，返回完整对读结果。"""
    m = len(base)
    names = [w["name"] for w in witnesses]
    per_witness = [_witness_readings(base, w["text"]) for w in witnesses]

    entries = []

    def add_entry(coordinate: dict, base_reading: str, base_span: list, readings: list):
        groups = {}
        witness_spans = {}
        for name, r in zip(names, readings):
            group = groups.setdefault(
                r["reading"],
                {"reading": r["reading"], "kind": r["kind"], "witnesses": []},
            )
            group["witnesses"].append(name)
            witness_spans[name] = [r["span"]]
        diff_kinds = sorted(
            {g["kind"] for g in groups.values() if g["kind"] != OP_MATCH}
        )
        entries.append(
            {
                "id": len(entries),
                "coordinate": coordinate,
                "base_reading": base_reading,
                "diff_kinds": diff_kinds,
                "groups": list(groups.values()),
                "spans": {"base": [base_span], "witnesses": witness_spans},
            }
        )

    # 依坐标顺序扫描：间隙0、第1字、间隙1、第2字、……、第m字、间隙m
    for type_, _idx, coordinate, base_reading, base_span in _iter_coordinates(base):
        if type_ == "gap":
            g = coordinate["index"]
            readings = [pw[1][g] for pw in per_witness]
        else:
            i = coordinate["index"] - 1
            readings = [pw[0][i] for pw in per_witness]
        if any(r["kind"] != OP_MATCH for r in readings):
            add_entry(coordinate, base_reading, base_span, readings)

    return {
        "base": {"text": base, "length": m},
        "witnesses": [
            {"name": w["name"], "text": w["text"], "length": len(w["text"])}
            for w in witnesses
        ],
        "entries": entries,
        "stats": {
            "base_length": m,
            "witness_count": len(witnesses),
            "diff_positions": len(entries),
        },
    }


def provenance(base: str, witnesses: list, query: str) -> dict:
    """待鉴文本来源分段：求总分最低的完整来源路径。

    复用 align／_witness_readings 的校验后对齐结果，把待鉴文本与各抄本的
    读法一并映射到底本字符／间隙坐标。

    鉴别坐标：各抄本读法不全相同的坐标（待鉴读法本身不参与判定）。
    只在鉴别坐标上为待鉴读法指派一份来源抄本：
      待鉴读法与来源抄本相同计 0，否则计 1；
      相邻鉴别坐标来源不同（换源）另计 2。
    求总分最低的完整来源路径；总分相同时先取换源次数少者，再按抄本录入
    顺序的来源序列取字典序最小者（以录入序号比较）。

    返回 {"decidable", "discriminative_count", "score", "switches",
    "mismatches", "path", "segments"}；无鉴别坐标时 decidable=False。
    segments 把路径按连续同源的鉴别坐标归并成段，给出首尾坐标、
    来源抄本名与段内失配数。
    """
    names = [w["name"] for w in witnesses]
    per_witness = [_witness_readings(base, w["text"]) for w in witnesses]
    q_char, q_gap = _witness_readings(base, query)

    coords = []  # (coordinate, witness_readings, query_reading)
    for _type, _idx, coordinate, _br, _span in _iter_coordinates(base):
        if coordinate["type"] == "gap":
            g = coordinate["index"]
            witness_readings = [pw[1][g]["reading"] for pw in per_witness]
            query_reading = q_gap[g]["reading"]
        else:
            i = coordinate["index"] - 1
            witness_readings = [pw[0][i]["reading"] for pw in per_witness]
            query_reading = q_char[i]["reading"]
        # 鉴别坐标仅看抄本读法是否不全相同
        if len(set(witness_readings)) >= 2:
            coords.append((coordinate, witness_readings, query_reading))

    if not coords:
        return {
            "decidable": False,
            "discriminative_count": 0,
            "score": None,
            "switches": 0,
            "mismatches": 0,
            "path": [],
            "segments": [],
        }

    nw = len(witnesses)
    k = len(coords)

    # 动态规划：states[t][s] 为覆盖前 t+1 个鉴别坐标、最末来源为第 s 份
    # 抄本时的最优状态 (总分, 换源次数, 来源序号序列)。同一末来源下被
    # 逐项压倒的状态无需保留——后续代价只取决于末来源，而追加相同后缀
    # 不改变总分、换源次数的大小关系与序列字典序。
    _c0, wr0, qr0 = coords[0]
    states = [[(0 if qr0 == wr0[s] else 1, 0, (s,)) for s in range(nw)]]
    for t in range(1, k):
        _coord, wr, qr = coords[t]
        add = [0 if qr == wr[s] else 1 for s in range(nw)]
        layer = []
        for s in range(nw):
            best = None
            for p in range(nw):
                pt, psw, pseq = states[t - 1][p]
                switched = p != s
                cand = (
                    pt + add[s] + (2 if switched else 0),
                    psw + (1 if switched else 0),
                    pseq + (s,),
                )
                # 元组比较即「总分 → 换源次数 → 来源序列字典序」
                if best is None or cand < best:
                    best = cand
            layer.append(best)
        states.append(layer)

    score, switches, seq = min(states[k - 1][s] for s in range(nw))

    path = []
    for t, (coordinate, wr, qr) in enumerate(coords):
        s = seq[t]
        path.append(
            {
                "coordinate": {
                    "type": coordinate["type"],
                    "index": coordinate["index"],
                    "label": coordinate["label"],
                },
                "query_reading": qr,
                "source": names[s],
                "mismatch": 0 if qr == wr[s] else 1,
                "witness_readings": [
                    {
                        "name": names[w],
                        "reading": wr[w],
                        "matches_query": wr[w] == qr,
                    }
                    for w in range(nw)
                ],
            }
        )

    segments = []
    start = 0
    for t in range(1, k + 1):
        if t == k or seq[t] != seq[start]:
            end = t - 1
            segments.append(
                {
                    "start": dict(path[start]["coordinate"]),
                    "end": dict(path[end]["coordinate"]),
                    "witness": names[seq[start]],
                    "mismatches": sum(path[u]["mismatch"] for u in range(start, t)),
                }
            )
            start = t

    return {
        "decidable": True,
        "discriminative_count": k,
        "score": score,
        "switches": switches,
        "mismatches": sum(p["mismatch"] for p in path),
        "path": path,
        "segments": segments,
    }


def _mst(nw: int, coords: list, active: list):
    """在指定鉴别坐标上构造抄本最小生成树。

    coords 为 (coordinate, 各抄本读法) 列表，active 为参与计距的下标。
    两份抄本的距离＝这些坐标上读法不同的次数；候选边按「距离升序、
    再按两端录入序号（i 小者在前、再比 j）升序」排序后做 Kruskal 选边，
    并列时唯一定边。返回 [(distance, i, j), …]（i < j，选边顺序）。
    """
    edges = []
    for i in range(nw):
        for j in range(i + 1, nw):
            distance = 0
            for t in active:
                readings = coords[t][1]
                if readings[i] != readings[j]:
                    distance += 1
            edges.append((distance, i, j))
    edges.sort()  # 元组排序即 (距离, i, j)

    parent = list(range(nw))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    chosen = []
    for distance, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            chosen.append((distance, i, j))
            if len(chosen) == nw - 1:
                break
    return chosen


def relationships(base: str, witnesses: list) -> dict:
    """抄本异文距离树：鉴别坐标上的抄本间距与最小生成树。

    复用 align／_witness_readings 的对齐结果，只取各抄本读法不全相同的
    底本字符位与间隙（鉴别坐标）。两份抄本的距离为这些坐标上读法不同的
    次数；按距离升序、两端录入序号升序 Kruskal 选边，得到连接全部抄本的
    最小生成树。再逐一剔除每个鉴别坐标重算树：原树每条边标出被保留次数
    与鉴别坐标总数（保留比例即二者之商），并附两端读法不同的坐标明细。

    返回 {"decidable", "discriminative_count", "tree"}；无鉴别坐标时
    decidable=False、tree 为空（无法比较，不生成树）。
    """
    names = [w["name"] for w in witnesses]
    nw = len(witnesses)
    per_witness = [_witness_readings(base, w["text"]) for w in witnesses]

    coords = []  # (coordinate, 各抄本读法)
    for _type, _idx, coordinate, _br, _span in _iter_coordinates(base):
        if coordinate["type"] == "gap":
            readings = [pw[1][coordinate["index"]]["reading"] for pw in per_witness]
        else:
            readings = [pw[0][coordinate["index"] - 1]["reading"] for pw in per_witness]
        if len(set(readings)) >= 2:
            coords.append((coordinate, readings))

    if not coords:
        return {"decidable": False, "discriminative_count": 0, "tree": []}

    k = len(coords)
    tree = _mst(nw, coords, list(range(k)))

    # 逐一剔除一个鉴别坐标重算树，登记原树各边是否仍在
    jackknife = []
    for skip in range(k):
        jackknife.append({(i, j) for _d, i, j in
                          _mst(nw, coords, [t for t in range(k) if t != skip])})

    tree_out = []
    for distance, i, j in tree:
        differences = []
        for coordinate, readings in coords:
            if readings[i] != readings[j]:
                differences.append(
                    {
                        "coordinate": {
                            "type": coordinate["type"],
                            "index": coordinate["index"],
                            "label": coordinate["label"],
                        },
                        "a_reading": readings[i],
                        "b_reading": readings[j],
                    }
                )
        tree_out.append(
            {
                "a": {"index": i + 1, "name": names[i]},
                "b": {"index": j + 1, "name": names[j]},
                "distance": distance,
                "retained": sum((i, j) in edges for edges in jackknife),
                "total": k,
                "differences": differences,
            }
        )

    return {"decidable": True, "discriminative_count": k, "tree": tree_out}


def alignment_audit(base: str, witnesses: list) -> dict:
    """最优对齐路径审计：逐抄本统计最小代价路径数并比较次小路径。

    对每份抄本，沿用 align 的编辑代价（匹配 0；替换／删除／插入 1）
    精确统计所有最小代价操作路径——不同操作序列算不同路径，路径数
    以十进制字符串返回（可能极大）。按「匹配＜替换＜删除＜插入」的
    操作序列字典序，首条路径即 align 的取舍结果；路径数大于 1 时
    另求严格次小的路径，比较两条路径在底本字符与间隙坐标上的读法
    及抄本码点半开区间，列出有差别的坐标及两侧值（读法相同但区间
    不同的坐标同样列出）。

    返回 {"witnesses": [{"name", "path_count", "alternative"}]}；
    alternative 为 None（路径唯一）或 {"differences": [...]}。
    """
    results = []
    for w in witnesses:
        text = w["text"]
        snap = _snapshot(base, text)
        cost = snap.cost
        first_events = snap.events
        count = _path_count(base, text, cost)
        alternative = None
        if count > 1:
            second_events = _second_events(base, text, cost, first_events)
            first_readings = snap.readings
            second_readings = _readings_from_events(base, text, second_events)
            differences = []
            for type_, idx, coordinate, _br, _span in _iter_coordinates(base):
                if type_ == "gap":
                    r_first = first_readings[1][idx]
                    r_second = second_readings[1][idx]
                else:
                    r_first = first_readings[0][idx - 1]
                    r_second = second_readings[0][idx - 1]
                if r_first != r_second:
                    differences.append(
                        {
                            "coordinate": coordinate,
                            "first": r_first,
                            "second": r_second,
                        }
                    )
            alternative = {"differences": differences}
        results.append(
            {
                "name": w["name"],
                "path_count": str(count),
                "alternative": alternative,
            }
        )
    return {"witnesses": results}


def distinguishing_window(base: str, witnesses: list) -> dict:
    """最短抄本鉴别片段：坐标数最少、能区分每一对抄本的连续坐标窗口。

    沿用 align／_witness_readings 的首条最优对齐，把全部坐标按
    「间隙0、第1字、间隙1、……、第m字、间隙m」排成一列。寻找连续坐标
    窗口，使每对抄本在窗口内至少有一处读法不同；取坐标数最少的窗口，
    并列时取起点最早者。若有抄本对在整篇均无不同读法，则无法鉴别，
    列出这些抄本对。

    返回 {"decidable", "coordinate_count", "window", "undistinguishable",
    "quotes"}：
      - 可鉴别时 window 含 start／end 坐标（type／index／label）与
        coordinate_count（窗口坐标数）；quotes 依窗口坐标顺序给出底本与
        各抄本的短引文：text 为窗口读法拼成的引文，items 逐坐标含
        coordinate、reading 与该读法在原文中的码点半开区间 span
        （[start, end)，空读法保留零长区间）。
      - 无法鉴别时 window 与 quotes 为空，undistinguishable 逐对列出
        整篇读法无异的抄本（a／b 含录入序号 index（1 起）与 name，
        a.index < b.index）。
    """
    names = [w["name"] for w in witnesses]
    nw = len(witnesses)
    per_witness = [_witness_readings(base, w["text"]) for w in witnesses]

    coords = []  # (coordinate, 底本读法, 底本码点区间, 各抄本读法)
    for _type, _idx, coordinate, base_reading, base_span in _iter_coordinates(base):
        if coordinate["type"] == "gap":
            readings = [pw[1][coordinate["index"]] for pw in per_witness]
        else:
            readings = [pw[0][coordinate["index"] - 1] for pw in per_witness]
        coords.append((coordinate, base_reading, base_span, readings))

    length = len(coords)
    # diff_at[t]：在坐标 t 上读法不同的抄本对（位掩码，位 i*nw+j 表示 (i, j)）
    diff_at = []
    for _coordinate, _br, _base_span, readings in coords:
        mask = 0
        for i in range(nw):
            for j in range(i + 1, nw):
                if readings[i]["reading"] != readings[j]["reading"]:
                    mask |= 1 << (i * nw + j)
        diff_at.append(mask)

    pair_count = nw * (nw - 1) // 2
    full_mask = 0
    for i in range(nw):
        for j in range(i + 1, nw):
            full_mask |= 1 << (i * nw + j)
    union_all = 0
    for mask in diff_at:
        union_all |= mask
    missing = full_mask & ~union_all
    if missing:
        pairs = []
        for i in range(nw):
            for j in range(i + 1, nw):
                if missing & (1 << (i * nw + j)):
                    pairs.append(
                        {
                            "a": {"index": i + 1, "name": names[i]},
                            "b": {"index": j + 1, "name": names[j]},
                        }
                    )
        return {
            "decidable": False,
            "coordinate_count": 0,
            "window": None,
            "undistinguishable": pairs,
            "quotes": [],
        }

    # 滑动窗口：对每个右端 right 求最短可行左端 left（覆盖全部抄本对）。
    # 坐标数严格更小才更新，故坐标数并列时自然保留起点最早者。
    best = None  # (left, right) 含端点
    left = 0
    covered = 0
    counts = [0] * (nw * nw)
    for right in range(length):
        mask = diff_at[right]
        bit = 0
        while mask:
            if mask & 1:
                counts[bit] += 1
                if counts[bit] == 1:
                    covered += 1
            mask >>= 1
            bit += 1
        while covered == pair_count:
            if best is None or right - left < best[1] - best[0]:
                best = (left, right)
            mask = diff_at[left]
            bit = 0
            while mask:
                if mask & 1:
                    counts[bit] -= 1
                    if counts[bit] == 0:
                        covered -= 1
                mask >>= 1
                bit += 1
            left += 1

    start, end = best
    window_coords = coords[start : end + 1]

    base_items = []
    pos = 0
    for coordinate, base_reading, base_span, _readings in window_coords:
        item_span = [pos, pos + len(base_reading)]
        pos += len(base_reading)
        base_items.append(
            {
                "coordinate": dict(coordinate),
                "reading": base_reading,
                "span": item_span,
                "original_span": list(base_span),
            }
        )

    quotes = [
        {
            "name": "底本",
            "text": "".join(item["reading"] for item in base_items),
            "items": base_items,
        }
    ]
    for w in range(nw):
        items = []
        pos = 0
        for coordinate, _br, _base_span, readings in window_coords:
            r = readings[w]
            item_span = [pos, pos + len(r["reading"])]
            pos += len(r["reading"])
            items.append(
                {
                    "coordinate": dict(coordinate),
                    "reading": r["reading"],
                    "span": item_span,
                    "original_span": list(r["span"]),
                }
            )
        quotes.append(
            {
                "name": names[w],
                "text": "".join(item["reading"] for item in items),
                "items": items,
            }
        )

    return {
        "decidable": True,
        "coordinate_count": end - start + 1,
        "window": {
            "start": dict(coords[start][0]),
            "end": dict(coords[end][0]),
            "coordinate_count": end - start + 1,
        },
        "undistinguishable": [],
        "quotes": quotes,
    }


def block_move(base: str, witnesses: list) -> dict:
    """单次片段搬移定位：逐抄本判断能否由底本一次连续片段搬移得到。

    搬移＝从底本剪出非空码点半开区间 [i, j)（j > i），在删除后的文本中
    选择不同于原位置的插入索引 k，再把原片段插回。只接受拼合结果与抄本
    逐码点完全相同的方案，空白与标点照常参与、不做任何规范化。

    长度不同必然无解；底本与抄本完全相同标为「未搬移」。存在多种方案时
    先取片段最短（j−i 最小），再取 i 最小、k 最小者（i＝k 表示原位置
    插回，不算搬移，已排除；同一 (L, i) 下向左的可行 k 恒小于向右者，
    各方向内用 find 自最小候选查起）。

    返回 {"witnesses": [{"name", "status", "plan"}]}：
      status 为 "unmoved"（未搬移）、"moved"（可单次搬移）或
      "impossible"（不能以单次搬移解释）；
      moved 时 plan 含 base_span（底本片段区间 [i, j)）、insert_index
      （删除后文本中的插入索引 k）、witness_span（片段在抄本中的新区间
      [k, k+L)）、fragment（片段文字）与 result_text（搬移后全文）。
    """
    results = []
    for w in witnesses:
        plan = _block_move_plan(base, w["text"])
        entry = {"name": w["name"], "status": plan[0], "plan": plan[1]}
        results.append(entry)
    return {"witnesses": results}


def _block_move_plan(base: str, text: str):
    """求单份抄本相对底本的最优单次搬移方案，返回 (status, plan)。

    设剪出片段为 base[i:i+L]（L≥1），残文 base[:i]+base[i+L:] 长度 m−L，
    插入索引 k∈[0, m−L] 且 k≠i。拼合结果在两种方向下分别为：
      向左（k<i）：base[:k] + base[i:i+L] + base[k:i] + base[i+L:]
      向右（k>i）：base[:i] + base[i+L:k+L] + base[i:i+L] + base[k+L:]
    片段在抄本中的新区间均为 [k, k+L)。记两串首处不同为 a、末处不同为 b：
      向左：前缀 base[:k] 不变须 k≤a；后缀 base[i+L:] 不变须 i+L>b；
            中段交换须 base[k:i]==text[k+L:i+L]；
            且 text[k:k+L]==base[i:i+L]。
      向右：前缀 base[:i] 不变须 i≤a；后缀 base[k+L:] 不变须 k+L>b；
            中段交换须 base[i+L:k+L]==text[i:k]；
            且 text[k:k+L]==base[i:i+L]。

    固定 L 时预计算两条错位对角上的连续相等段（各 O(m)）：
      run_r[i]：base[i+L+t]==text[i+t] 自 t=0 起的连续成立数；
      run_l_end[i]：base[p]==text[p+L] 在 p=i−1,i−2,… 上的连续成立数。
    可行 k 用 str.find 沿抄本定位片段出现点，配合上述界与相等段在 O(1)
    内判定；命中后仍直接拼合复核逐码点相等。枚举顺序 L↑、i↑，同一
    (L, i) 下向左的 k 恒小于向右的 k，故先左后右——即「片段最短 →
    i 最小 → k 最小」的取舍顺序。
    """
    m = len(base)
    if len(text) != m:
        return "impossible", None
    if text == base:
        return "unmoved", None

    # 首、末处不同的码点位置（两串不同且等长，故 0 ≤ a ≤ b < m）
    a = 0
    while base[a] == text[a]:
        a += 1
    b = m - 1
    while base[b] == text[b]:
        b -= 1

    for L in range(1, m):  # L＝m 时残文无其他间隙，不存在 k≠i
        # run_r[i]：自 (base i+L, text i) 起沿对角的最长相等段
        run_r = [0] * (m - L + 1)
        for i in range(m - L - 1, -1, -1):
            if base[i + L] == text[i]:
                run_r[i] = 1 + run_r[i + 1]
        # run_l_end[i]：错位 L 的对角上恰在位置 i 之前的连续相等段长度
        run_l_end = [0] * (m - L + 1)
        for i in range(1, m - L + 1):
            if base[i - 1] == text[i - 1 + L]:
                run_l_end[i] = run_l_end[i - 1] + 1
        for i in range(0, m - L + 1):
            frag = base[i : i + L]
            # 向左：k∈[0, i−1]，k≤a，i+L>b，中段自 r 起须全等
            if i >= 1 and i + L > b:
                r = i - run_l_end[i]
                k = text.find(frag, r, i + L)
                if k != -1 and k < i and k <= a:
                    plan = _make_move_plan(base, i, i + L, k, L)
                    if plan["result_text"] == text:
                        return "moved", plan
            # 向右：k∈[i+1, m−L]，i≤a，k+L>b，中段自 i 起须全等
            if i <= a and i < m - L:
                lo = max(i + 1, b + 1 - L)
                k = text.find(frag, lo)
                if k != -1 and k <= i + run_r[i]:
                    plan = _make_move_plan(base, i, i + L, k, L)
                    if plan["result_text"] == text:
                        return "moved", plan
    return "impossible", None


def _make_move_plan(base: str, i: int, j: int, k: int, L: int) -> dict:
    """组装单次搬移方案（[i,j) 剪出于残文索引 k 处插回）。"""
    fragment = base[i:j]
    residual = base[:i] + base[j:]
    result = residual[:k] + fragment + residual[k:]
    return {
        "base_span": [i, j],
        "insert_index": k,
        "witness_span": [k, k + L],
        "fragment": fragment,
        "result_text": result,
    }


def _coord_desc(type_: str, index: int, m: int) -> str:
    if type_ == "char":
        return f"第{index}字"
    if index == 0:
        return "文首间隙（间隙0）"
    if index == m:
        return f"文末间隙（间隙{m}）"
    return f"第{index}字后间隙（间隙{index}）"


def _reading_prompt(reading: str, type_: str) -> str:
    if reading != "":
        return f"读法「{reading}」"
    return "缺字" if type_ == "char" else "无插入"


def edition(base: str, witnesses: list, selections: list) -> dict:
    """依校定选择拼成完整校定文本。

    selections 仅记录非默认选择，每条为 {"type", "index", "reading"}：
      type 为 "char"（字符坐标）或 "gap"（间隙坐标）；
      char 的 index 为 1…m，gap 的 index 为 0…m；
      reading 为底本读法或该坐标下某份抄本的实际读法。
    默认取底本读法：字符坐标保留底本该字，间隙坐标不插入任何文字。

    返回 {"text", "length", "coordinates"}：coordinates 依拼字顺序
    列出全部坐标，含所选读法、该读法在成稿中的 Unicode 码点半开区间
    [start, end) 及支持该读法的抄本名（底本读法附底本；缺字／无插入
    保留零长区间）。
    """
    m = len(base)
    names = [w["name"] for w in witnesses]
    per_witness = [_witness_readings(base, w["text"]) for w in witnesses]

    # 仅差异坐标允许出现在 selections 中：其余坐标不存在异文条目，
    # 底本读法是唯一可选读法。
    diffs = {}  # (type, index) -> {"base": 底本读法, "supports": {读法: [抄本名]}}
    for type_, idx, coordinate, base_reading, _span in _iter_coordinates(base):
        if type_ == "gap":
            rs = [pw[1][coordinate["index"]] for pw in per_witness]
        else:
            rs = [pw[0][coordinate["index"] - 1] for pw in per_witness]
        if any(r["kind"] != OP_MATCH for r in rs):
            supports = {}
            for name, r in zip(names, rs):
                supports.setdefault(r["reading"], []).append(name)
            diffs[(type_, idx)] = {"base": base_reading, "supports": supports}

    chosen = {}
    for sel in selections:
        key = (sel["type"], sel["index"])
        if key in chosen:
            raise EditionError(f"坐标重复：{_coord_desc(key[0], key[1], m)}")
        type_, index = key
        if type_ not in ("char", "gap"):
            raise EditionError(f"无效坐标类型：{type_}（仅支持 char／gap）")
        if type_ == "char" and not 1 <= index <= m:
            raise EditionError(f"无效字符坐标：第{index}字（底本共 {m} 字）")
        if type_ == "gap" and not 0 <= index <= m:
            raise EditionError(f"无效间隙坐标：间隙{index}（合法范围 0～{m}）")
        if key not in diffs:
            raise EditionError(f"{_coord_desc(type_, index, m)}不是差异条目，无可选读法")
        diff = diffs[key]
        if sel["reading"] not in diff["supports"]:
            raise EditionError(
                f"{_coord_desc(type_, index, m)}不存在{_reading_prompt(sel['reading'], type_)}，"
                "该读法不属于此坐标"
            )
        chosen[key] = sel["reading"]

    parts = []
    coordinates = []
    pos = 0
    for type_, idx, coordinate, base_reading, _span in _iter_coordinates(base):
        if (type_, idx) in chosen:
            reading = chosen[(type_, idx)]
            if (type_, idx) in diffs and reading == diffs[(type_, idx)]["base"]:
                supporters = ["底本", *diffs[(type_, idx)]["supports"][reading]]
            else:
                supporters = list(diffs[(type_, idx)]["supports"][reading])
            is_default = False
        else:
            reading = base_reading
            supporters = ["底本"]
            is_default = True
        start = pos
        pos += len(reading)
        coordinates.append(
            {
                "coordinate": {"type": coordinate["type"], "index": coordinate["index"],
                               "label": coordinate["label"]},
                "reading": reading,
                "default": is_default,
                "span": [start, pos],
                "supporters": supporters,
            }
        )
        parts.append(reading)

    return {
        "text": "".join(parts),
        "length": pos,
        "coordinates": coordinates,
    }
