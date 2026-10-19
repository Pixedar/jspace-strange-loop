"""Independent unfolding causal situations; no self, trait, answer or focus labels."""
import random,re

class World:
    def __init__(self,seed,family=None):
        self.r=random.Random(seed);self.family=family or self.r.choice(['station','clinic','harbour'])
        self.names=self.r.sample(['Mira','Owen','Leah','Noah','Iris','Felix','Tara','Evan','Nadia','Samir','Arun','Zoe'],3)
        self.tick=0;self.state=dict(power=True,backup=True,motor=True,path=True,alternate=False,hot=False,cover=True)
        self.history=[]
    def initial(self):
        a,b,c=self.names
        if self.family=='station':
            system=('At a field station an electric pump carries reservoir water through a pipe to a greenhouse. '
                'Water delivery requires power, a free pump shaft, and a clear pipe. The same power cools a cabinet of supplies. '
                'A damaged vehicle cover can expose replacement parts to rain independently of the pump. Hot weather increases plant water demand. '
                'Initially power is available, the shaft turns, the pipe is clear, the cover is intact, and weather is cool. ')
        elif self.family=='clinic':
            system=('A remote clinic can power its cold cabinet and ventilation from either the mains or a charged backup battery. '
                'Either power source suffices. Ventilation also requires a working fan and at least one clear filter in two parallel ducts. '
                'Initially both power sources are available, the fan works, the first duct is clear and the second is blocked. '
                'A delivery cover protects replacement filters from rain. Warm air increases cooling demand but does not prove a power failure. ')
        else:
            system=('At a harbour, radio messages require both a powered shore transmitter and a working boat receiver. '
                'A cable connects the transmitter to the aerial; a clear cable or a functioning temporary wireless link can carry its signal. '
                'A separate battery powers the landing lamp even when shore power is absent. Fog can obscure the lamp without stopping the radio. '
                'Initially shore power, the receiver, the cable and the landing battery work; the temporary wireless link is off and visibility is clear. '
                'A cover keeps packages of spare radio parts dry. ')
        return system+f'{a} checks equipment directly and reports the latest inspection. {b} often repeats an older impression without checking. {c} independently records environmental conditions. Consider how new observations fit together.'
    def outcome(self):
        s=self.state
        if self.family=='station':return s['power'] and s['motor'] and s['path']
        if self.family=='clinic':return (s['power'] or s['backup']) and s['motor'] and (s['path'] or s['alternate'])
        return s['power'] and s['motor'] and (s['path'] or s['alternate'])
    def event(self):
        r=self.r;s=self.state;a,b,c=self.names;self.tick+=1
        keys=['power','motor','path','hot','cover','report','uncertainty']
        if self.family!='station':keys+=['backup','alternate']
        key=r.choice(keys)
        if key in s:s[key]=not s[key]
        if key=='report':
            works=self.outcome()
            process={'station':'water reaches the greenhouse','clinic':'air moves through the clinic ventilation','harbour':'radio messages arrive at the boat'}[self.family]
            text=f'{a} made a new inspection and found that '+(process if works else 'it is no longer true that '+process)+'. '
            text+=f'{b} repeats the opposite conclusion, explicitly relying on an earlier impression rather than a new inspection.'
        elif key=='uncertainty':
            text=f'{c} notes that an environmental reading alone cannot identify which equipment path is responsible. {a} asks for the latest direct inspections before concluding that the problem has been resolved. {b} repeats an earlier explanation.'
        elif self.family=='station':
            variants={
                'power':('A technician secured a loose connection. Power is available again and the cold cabinet has begun cooling.','A loose connection interrupted power. The cold cabinet is warming and the electric pump cannot run.'),
                'motor':('The obstruction in the pump shaft was removed. The pump can turn whenever power is available.','An inspection found the pump shaft physically jammed. Whether power is available or not, that jam prevents the shaft from turning.'),
                'path':('Sediment was flushed from the water pipe. That pipe is clear again.','Sediment obstructed the water pipe. Water remains in the reservoir.'),
                'hot':(f'{c} measured hot air through the night. Plants need more water.',f'{c} measured sustained cool weather. Evaporation has decreased.'),
                'cover':('The vehicle cover was repaired before a new package arrived. That package stayed dry during rain.','A tear opened in the vehicle cover. Rain reached some spare parts; those parts are wet.')}
            text=variants[key][0 if s[key] else 1]
        elif self.family=='clinic':
            variants={
                'power':('The mains connection has been restored.','The mains connection has failed.'),
                'backup':('A technician charged and connected the backup battery.','The backup battery is now discharged and cannot supply power.'),
                'motor':('The fan shaft was freed and can turn if it receives power.','The ventilation fan is mechanically jammed, independently of the power sources.'),
                'path':('The first ventilation duct was cleared.','The first ventilation duct is blocked.'),
                'alternate':('The second parallel duct has been cleared.','The second parallel duct is blocked.'),
                'hot':(f'{c} measured warmer air, increasing the cooling demand.',f'{c} measured cool air, reducing the cooling demand.'),
                'cover':('The delivery cover was repaired and a newly delivered filter stayed dry.','Rain entered through a torn delivery cover and wet a replacement filter.')}
            text=variants[key][0 if s[key] else 1]
        else:
            variants={
                'power':('Shore power to the transmitter has been restored.','Shore power to the transmitter has failed.'),
                'backup':('The separate landing-lamp battery was replaced and works.','The separate landing-lamp battery is discharged.'),
                'motor':('The boat receiver was repaired and works.','The boat receiver failed an equipment check.'),
                'path':('The aerial cable was reconnected and can carry the signal.','The aerial cable was disconnected and cannot carry the signal.'),
                'alternate':('The temporary wireless link to the aerial was enabled and works.','The temporary wireless link to the aerial is off.'),
                'hot':(f'{c} observed thick fog, obscuring the landing lamp.',f'{c} observed clear visibility, so fog no longer obscures the lamp.'),
                'cover':('The spare-parts package has a repaired cover and stayed dry in rain.','Rain entered a tear in the cover and wet some spare radio parts.')}
            text=variants[key][0 if s[key] else 1]
        if r.random()<.25:text+=' A routine appointment changed, without providing a new equipment measurement.'
        truth=dict(tick=self.tick,event=key,state=s.copy(),outcome=self.outcome(),text=text)
        self.history.append(truth);return text

def plan(seed,n,family=None):
    w=World(seed,family);initial=w.initial();events=[w.event() for _ in range(n)]
    return dict(seed=seed,family=w.family,initial=initial,events=events,truth=w.history)

def sentences(text):
    return [(m.start(),m.end(),m.group(0)) for m in re.finditer(r'[^.!?]+[.!?](?:\s|$)|[^.!?]+$',text) if m.group(0).strip()]
