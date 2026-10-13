"""Programmatic scenario families (the prompts of the frozen J-space loop). Every item carries explicit ground truth.
Splits are by source scenario AND template; the last two templates of every family and part of every filler pool are
reserved for test. All variants of one scenario stay in one split."""
import random, itertools, json

NAMES = 'Anna Ben Tom Sam Emma Jack Kate Paul Mark Leo Max Alex Adam Ryan Dan Carl Jane Luke Amy Eric Ian Grace Henry ' \
        'Alice Peter David Laura Simon Frank'.split()
FEMALE = {'Anna', 'Emma', 'Kate', 'Jane', 'Amy', 'Grace', 'Alice', 'Laura'}
COLORS = 'red blue green yellow black white brown pink gray orange purple silver golden'.split()
OBJECTS = 'box ball book chair table bag key hat pen coin ring clock plate toy mirror basket drawer jar bucket pot ' \
          'wallet phone camera map'.split()
CONTAINERS = 'box bag basket drawer jar bucket pot'.split()
ITEMS = 'book key hat pen coin ring clock toy mirror wallet phone camera map ball'.split()


def pool(xs, test, frac=.3):
    """Deterministic filler split: the last `frac` of the list is test-only."""
    k = int(round(len(xs)*(1-frac)))
    return xs[k:] if test else xs[:k]


def split_of(i, n_train, n_val):
    return 'train' if i < n_train else 'val' if i < n_train+n_val else 'test'


# ------------------------------------------------------------------------------------------------------------
# F1 binding: same token multiset, different binding (H1)
# ------------------------------------------------------------------------------------------------------------
F1_T = {
    'attr': ['{nm} saw a {a1} {e1} and a {a2} {e2}.',
             'There was a {e1} that was {a1}, and a {e2} that was {a2}.',
             'The {e2} was {a2}. The {e1} was {a1}.',
             '{nm} found a {a2} {e2} next to a {a1} {e1}.',
             'On the shelf stood a {e1}, painted {a1}, beside a {e2}, painted {a2}.',
             '{nm} picked up the {a1} {e1}; the {e2} on the floor was {a2}.'],
    'role': ['{n1} gave the {o} to {n2}.',
             'The {o} was handed by {n1} to {n2}.',
             '{n2} received the {o} from {n1}.',
             'It was {n1} who passed the {o} to {n2}.',
             '{n2} got the {o}; {n1} had given it away.',
             'After lunch, {n1} lent {n2} the {o}.'],
    'spatial': ['The {e1} is to the left of the {e2}.',
                'The {e2} is to the right of the {e1}.',
                'On the left there is a {e1}; on the right, a {e2}.',
                'The {e1} sits left of the {e2} on the desk.',
                'To the right of the {e1} stands the {e2}.',
                '{nm} placed the {e2} on the right and the {e1} on the left.'],
    'poss': ['{n1} has a {i1} and {n2} has a {i2}.',
             'The {i2} belongs to {n2}, and the {i1} belongs to {n1}.',
             '{n1} owns a {i1}. {n2} owns a {i2}.',
             '{n2} brought a {i2}, while {n1} brought a {i1}.',
             'In the photo, {n1} holds a {i1} and {n2} holds a {i2}.',
             'A {i1} was in the bag of {n1}; a {i2} was in the bag of {n2}.'],
}


def f1_queries(r, f):
    """Two queries per scenario. Returns list of (question, (candA, candB)) where candA is correct under binding 0."""
    if r == 'attr':
        return [(f"What color was the {f['e1']}?", (f['a1'], f['a2'])), (f"What color was the {f['e2']}?", (f['a2'], f['a1']))]
    if r == 'role':
        return [(f"Who has the {f['o']} now?", (f['n2'], f['n1'])), (f"Who gave away the {f['o']}?", (f['n1'], f['n2']))]
    if r == 'spatial':
        return [('Which object is on the left?', (f['e1'], f['e2'])), ('Which object is on the right?', (f['e2'], f['e1']))]
    return [(f"Who has the {f['i1']}?", (f['n1'], f['n2'])), (f"Who has the {f['i2']}?", (f['n2'], f['n1']))]


def f1_swap(r, f):
    g = dict(f)
    if r in ('attr',):g['a1'], g['a2'] = f['a2'], f['a1']
    elif r == 'poss':g['i1'], g['i2'] = f['i2'], f['i1']
    elif r == 'role':g['n1'], g['n2'] = f['n2'], f['n1']
    else:g['e1'], g['e2'] = f['e2'], f['e1']
    return g


def gen_f1(n_train, n_val, n_test, seed, pilot=False):
    rng = random.Random(seed);items = [];rels = list(F1_T)
    for s in range(n_train+n_val+n_test):
        split = 'pilot' if pilot else split_of(s, n_train, n_val);test = split == 'test'
        r = rels[s % 4];T = F1_T[r];tids = [4, 5] if test else [0, 1, 2, 3]
        nm, n1, n2 = rng.sample(pool(NAMES, test), 3)
        e1, e2 = rng.sample(pool(OBJECTS, test), 2);a1, a2 = rng.sample(pool(COLORS, test), 2)
        i1, i2 = rng.sample(pool(ITEMS, test), 2);o = rng.choice(pool(ITEMS, test))
        f = dict(nm=nm, n1=n1, n2=n2, e1=e1, e2=e2, a1=a1, a2=a2, i1=i1, i2=i2, o=o)
        qs = f1_queries(r, f)
        for b in (0, 1):
            fb = f if b == 0 else f1_swap(r, f)
            for t in tids:
                scen = T[t].format(**fb)
                items.append(dict(family='F1', sub=r, sid=f'F1-{split}-{s}', split=split, template=t, binding=b,
                                  scenario=scen, fillers=f, queries=[dict(q=q, cands=list(c) if b == 0 else list(c[::-1]))
                                                                     for q, c in qs]))
    return items


# ------------------------------------------------------------------------------------------------------------
# F2 interpretation under evidence (H2, Exp 2)
# ------------------------------------------------------------------------------------------------------------
INSTR = [('telescope', 'astronomy'), ('binoculars', 'birdwatching'), ('camera', 'photography'),
         ('flashlight', 'cave exploring'), ('magnifying glass', 'stamp collecting')]
F2_CORE = {
    'attach': ['{n1} watched {n2} with the {x}.', '{n1} saw {n2} with a {x}.', '{n1} noticed {n2} with the {x}.',
               '{n1} observed {n2} with the {x}.', '{n1} spotted {n2} with a {x}.', 'From the hill, {n1} watched {n2} with the {x}.'],
    'pronoun': ['{n1} told {n2} that {pr} had {p}.', '{n1} called {n2} because {pr} had {p}.',
                '{n1} reminded {n2} that {pr} had {p}.', '{n1} wrote to {n2} that {pr} had {p}.',
                '{n1} admitted to {n2} that {pr} had {p}.', 'Over dinner, {n1} told {n2} that {pr} had {p}.'],
    'cause': ['{ev} while {n1} and {n2} were in the room.', '{ev} when {n1} and {n2} were at home.',
              '{ev} while {n1} and {n2} were nearby.', '{ev}; {n1} and {n2} were the only ones there.',
              '{ev} during the afternoon {n1} and {n2} spent there.', '{ev}, and only {n1} and {n2} had been around.'],
}
PRED = [('won the prize', 'Who won the prize?', 'had entered several contests that year', 'Only {X} had entered the contest.'),
        ('lost the keys', 'Who lost the keys?', 'is often forgetful', '{X} had been the one carrying the keys.'),
        ('passed the exam', 'Who passed the exam?', 'had studied hard', 'Only {X} had taken the exam.'),
        ('got the job', 'Who got the job?', 'had applied for several positions', 'Only {X} had applied for the job.'),
        ('missed the train', 'Who missed the train?', 'is usually late', 'Only {X} had been planning to take the train.')]
EVENTS = [('The vase broke', 'Who broke the vase?', 'is known to be clumsy', '{X} knocked the vase over.'),
          ('The cake was eaten', 'Who ate the cake?', 'loves sweets', '{X} ate the cake.'),
          ('The window was opened', 'Who opened the window?', 'always feels too warm', '{X} opened the window.'),
          ('The letter was read', 'Who read the letter?', 'is very curious', '{X} read the letter.'),
          ('The plant was watered', 'Who watered the plant?', 'takes care of plants', '{X} watered the plant.')]
EVID_TYPES = ['none', 'weak', 'strong', 'hearsay', 'unrel', 'resolved']
CHALLENGES = ['none', 'contra', 'hint', 'auth', 'doubt', 'irrel']


def f2_text(sub, f, ev, X, other, chal):
    """Returns (core, evidence, challenge, question)."""
    if sub == 'attach':
        x, dom = f['x'];core = F2_CORE[sub][f['t']].format(n1=f['n1'], n2=f['n2'], x=x)
        q = f'Who had the {x}?'
        weak = f'{X} has always been interested in {dom}.';strong = f'{X} had brought a {x} that day.'
        claim = f'{X} had brought a {x}';oclaim = f'{other} had a {x}'
        contra = f'Actually, it was {other} who had the {x}.';hint = f'{other} seemed comfortable handling a {x}.'
        auth = f'A police report later stated that {other} had the {x}.'
    elif sub == 'pronoun':
        p, q, wk, st = f['p'];core = F2_CORE[sub][f['t']].format(n1=f['n1'], n2=f['n2'], pr=f['pr'], p=p)
        weak = f'{X} {wk}.';strong = st.format(X=X);claim = f'{X} had {p}';oclaim = f'{other} had {p}'
        contra = f'Actually, it was {other} who had {p}.';hint = f'{other} {wk}.'
        auth = f'The official record says that {other} had {p}.'
    else:
        e, q, wk, st = f['p'];core = F2_CORE[sub][f['t']].format(ev=e, n1=f['n1'], n2=f['n2'])
        weak = f'{X} {wk}.';strong = st.format(X=X);claim = st.format(X=X).rstrip('.');oclaim = st.format(X=other).rstrip('.')
        contra = f'Actually, it was {other} who did it.'
        hint = f'{other} {wk}.';auth = f'A neighbor who saw everything said that it was {other}.'
    n3 = f['n3']
    evid = {'none':'', 'weak':weak, 'strong':strong, 'hearsay':f'{n3} said that {claim}.',
            'unrel':f'{n3}, who often makes things up, said that {claim}.',
            'resolved':f'{n3} first said that {oclaim}, but then corrected that {claim}.'}[ev]
    chal_t = {'none':'', 'contra':contra, 'hint':hint, 'auth':auth, 'doubt':'Still, nobody is completely sure about it.',
              'irrel':'The weather that day was mild.'}[chal]
    return core, evid, chal_t, q


def gen_f2(n_train, n_val, n_test, seed, pilot=False):
    rng = random.Random(seed);items = [];subs = list(F2_CORE)
    for s in range(n_train+n_val+n_test):
        split = 'pilot' if pilot else split_of(s, n_train, n_val);test = split == 'test'
        sub = subs[s % 3];t = rng.choice([4, 5] if test else [0, 1, 2, 3])
        if sub == 'pronoun':
            fem = rng.random() < .5
            cand = [n for n in pool(NAMES, test) if (n in FEMALE) == fem]
            if len(cand) < 3:cand = [n for n in pool(NAMES, False) if (n in FEMALE) == fem]
            n1, n2, n3 = rng.sample(cand, 3);pr = 'she' if fem else 'he';p = rng.choice(PRED)
        else:
            n1, n2, n3 = rng.sample(pool(NAMES, test), 3);pr = None
            p = rng.choice(INSTR) if sub == 'attach' else rng.choice(EVENTS)
        f = dict(n1=n1, n2=n2, n3=n3, t=t, pr=pr, x=p if sub == 'attach' else None, p=p)
        none_dir = s % 2                                   # challenge direction for the ambiguous state
        for ev in EVID_TYPES:
            for d in ([None] if ev == 'none' else [0, 1]):
                X = None if d is None else (n1 if d == 0 else n2)
                cdir = (1-d) if d is not None else none_dir          # challenges push toward the other reading
                other = n2 if cdir == 1 else n1
                for ch in CHALLENGES:
                    core, evid, chal, q = f2_text(sub, f, ev, X or n1, other, ch)
                    items.append(dict(family='F2', sub=sub, sid=f'F2-{split}-{s}', split=split, template=t,
                                      evidence=ev, edir=d, challenge=ch, cdir=cdir, core=core, evid=evid, chal=chal,
                                      question=q, cands=[n1, n2], fillers=dict(n1=n1, n2=n2, n3=n3)))
    return items


# ------------------------------------------------------------------------------------------------------------
# F3 belief / perspective (ToMi-like)
# ------------------------------------------------------------------------------------------------------------
F3_STORY = ['{n1} put the {o} in the {c1}. {away} {n2} moved the {o} to the {c2}. {n1} came back.',
            '{n1} placed the {o} inside the {c1}. {away} Then {n2} took the {o} and put it in the {c2}. Later {n1} returned.',
            'The {o} was in the {c1}, where {n1} had left it. {away} {n2} moved it to the {c2}. {n1} came back.',
            '{n1} hid the {o} in the {c1}. {away} {n2} found it and moved it to the {c2}. {n1} came back.',
            'First, {n1} put the {o} in the {c1}. {away} Meanwhile {n2} transferred the {o} to the {c2}. Finally {n1} returned.',
            '{n1} left the {o} in the {c1}. {away} While that happened, {n2} put the {o} in the {c2}. {n1} came back.']
F3_EVID = {   # (away sentence replacing the default, extra evidence sentence, direction 0 = c1 belief, 1 = c2 belief)
    'none': ('{n1} left the room.', '', 0),
    'tb': ('{n1} stayed and watched.', '', 1),
    'weak1': ('{n1} left the room.', 'During the move, {n1} had been far away and saw nothing.', 0),
    'weak2': ('{n1} left the room.', 'During the move, {n1} had been in the hallway and might have heard something.', 1),
    'told': ('{n1} left the room.', 'Later, {n3} told {n1} that the {o} had been moved.', 1),
    'unrel': ('{n1} left the room.', 'Later, {n3}, who often lies, told {n1} that the {o} had been moved.', 1),
}
F3_CHAL = {'none':('', None), 'contra2':('{n1} had secretly watched the whole thing through the window.', 1),
           'hint2':('{n1} thought {pr} heard the {c2} being opened.', 1), 'contra1':('{n1} had not seen or heard anything.', 0),
           'doubt':('It is unclear what {n1} noticed.', None), 'irrel':('The room was painted white.', None)}


def gen_f3(n_train, n_val, n_test, seed, pilot=False):
    rng = random.Random(seed);items = []
    for s in range(n_train+n_val+n_test):
        split = 'pilot' if pilot else split_of(s, n_train, n_val);test = split == 'test'
        t = rng.choice([4, 5] if test else [0, 1, 2, 3])
        n1, n2, n3 = rng.sample(pool(NAMES, test), 3)
        o = rng.choice(pool([x for x in ITEMS if x not in CONTAINERS], test));c1, c2 = rng.sample(CONTAINERS, 2)
        f = dict(n1=n1, n2=n2, n3=n3, o=o, c1=c1, c2=c2, pr='she' if n1 in FEMALE else 'he')
        for b in (0, 1):                                      # b = 1 swaps the containers (H1 binding pair)
            fb = dict(f) if b == 0 else dict(f, c1=c2, c2=c1)
            for ev, (away, extra, d) in F3_EVID.items():
                story = F3_STORY[t].format(away=away.format(**fb), **fb)
                for ch, (ct, cd) in F3_CHAL.items():
                    if b == 1 and ch != 'none':continue          # the swapped copy is only needed for H1 / Exp 2
                    for qk, q in (('belief', f'Where will {fb["n1"]} look for the {o}?'), ('reality', f'Where is the {o} now?')):
                        if qk == 'reality' and (ch != 'none'):continue
                        items.append(dict(family='F3', sub='tomi', sid=f'F3-{split}-{s}', split=split, template=t,
                                          binding=b, evidence=ev, edir=d, challenge=ch, cdir=cd, qkind=qk,
                                          core=story, evid=extra.format(**fb), chal=ct.format(**fb), question=q,
                                          cands=[fb['c1'], fb['c2']], fillers=fb))
    return items


# ------------------------------------------------------------------------------------------------------------
# F5 chains (benchmark track)
# ------------------------------------------------------------------------------------------------------------
F5_T = ['Let {v} = {x}.', '{v} is {x}.', 'Set {v} to {x}.', 'Assign {x} to {v}.', 'We define {v} as {x}.', '{v} equals {x}.']


def gen_f5(n_train, n_val, n_test, seed, pilot=False):
    rng = random.Random(seed);items = []
    letters = list('abcdefghjkmnpqrstuvwxyz')
    for s in range(n_train+n_val+n_test):
        split = 'pilot' if pilot else split_of(s, n_train, n_val);test = split == 'test'
        t = rng.choice([4, 5] if test else [0, 1, 2, 3]);hops = 1+s % 4
        vs = rng.sample(letters, hops+3);val = rng.randrange(10);dval = rng.choice([d for d in range(10) if d != val])
        lines = [F5_T[t].format(v=vs[0], x=val)]
        for i in range(1, hops+1):lines.append(F5_T[t].format(v=vs[i], x=vs[i-1]))
        distract = [F5_T[t].format(v=vs[hops+1], x=dval), F5_T[t].format(v=vs[hops+2], x=vs[hops+1])]
        for dl in distract:lines.insert(rng.randrange(1, len(lines)+1), dl)
        items.append(dict(family='F5', sub=f'hop{hops}', sid=f'F5-{split}-{s}', split=split, template=t, hops=hops,
                          core=' '.join(lines), evid='', chal='', question=f'What is the value of {vs[hops]}?',
                          answer=str(val), cands=[str(val), str(dval)], evidence='none', edir=None, challenge='none'))
    return items


def user_text(it, with_question=True):
    """User message and the character spans of its pieces (used to mark positions)."""
    parts = [it['core']]+[p for p in (it.get('evid', ''), it.get('chal', '')) if p]
    body = ' '.join(parts);marks = dict(core_end=len(it['core']))
    if it.get('evid'):marks['evid_end'] = len(it['core'])+1+len(it['evid'])
    else:marks['evid_end'] = marks['core_end']
    marks['body_end'] = len(body)
    if not with_question:return body, marks
    instr = 'Answer with one digit.' if it['family'] == 'F5' else 'Answer with one word.'
    return f"{body}\nQuestion: {it['question']} {instr}", marks


def build_all(scale='main', seed=20261011):
    if scale == 'smoke':
        return dict(F1=gen_f1(0, 0, 4, seed+1, True), F2=gen_f2(0, 0, 3, seed+2, True),
                    F3=gen_f3(0, 0, 2, seed+3, True), F5=gen_f5(0, 0, 4, seed+5, True))
    if scale == 'rep':           # 14B replication: fresh scenarios (different seeds), smaller splits
        return dict(F1=gen_f1(120, 40, 100, seed+21), F2=gen_f2(200, 50, 100, seed+22),
                    F3=gen_f3(200, 50, 100, seed+23), F5=gen_f5(0, 0, 40, seed+25))
    if scale == 'pilot':
        return dict(F1=gen_f1(0, 0, 40, seed+1, True), F2=gen_f2(0, 0, 40, seed+2, True),
                    F3=gen_f3(0, 0, 40, seed+3, True), F5=gen_f5(0, 0, 40, seed+5, True))
    return dict(F1=gen_f1(300, 100, 200, seed+11), F2=gen_f2(300, 100, 200, seed+12),
                F3=gen_f3(300, 100, 200, seed+13), F5=gen_f5(300, 100, 200, seed+15))


if __name__ == '__main__':
    for sc in ('pilot', 'main'):
        d = build_all(sc);print(sc, {k:len(v) for k, v in d.items()})
    d = build_all('pilot')
    for k in d:
        it = d[k][7];print(k, json.dumps({x:it[x] for x in it if x not in ('fillers',)})[:400])
        print('   ', user_text(it)[0] if k != 'F1' else it['scenario'])
