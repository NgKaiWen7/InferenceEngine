from pathlib import Path

svg = r'''<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="900" viewBox="0 0 1400 900">
<rect width="100%" height="100%" fill="white"/>

<style>
.title {font-family:"DejaVu Sans",sans-serif; fill:#2348c0; font-size:32px; font-weight:bold;}
.txt {font-family:"DejaVu Sans",sans-serif; fill:#2348c0; font-size:26px;}
.red {stroke:#f04b4b; fill:none; stroke-width:5; stroke-linecap:round; stroke-linejoin:round;}
.blue {stroke:#2348c0; fill:none; stroke-width:5; stroke-linecap:round; stroke-linejoin:round;}
.orange {fill:#e98b2a; font-family:"DejaVu Sans",sans-serif; font-size:28px;}
</style>

<!-- Database -->
<text x="100" y="80" class="title">Database</text>
<!-- database cylinder icon -->
<ellipse cx="180" cy="180" rx="90" ry="35" class="blue"/>
<path d="M90 180 V400 C90 450 270 450 270 400 V180" class="blue"/>
<ellipse cx="180" cy="400" rx="90" ry="35" class="blue"/>
<text x="130" y="300" class="txt">Data</text>

<!-- Engine -->
<text x="420" y="80" class="title">Engine</text>
<!-- gear icon -->
<circle cx="500" cy="260" r="70" class="blue"/>
<circle cx="500" cy="260" r="25" class="blue"/>
<path d="M500 170 V130 M500 350 V390 M410 260 H370 M590 260 H630
M435 195 L405 165 M565 325 L595 355 M565 195 L595 165 M435 325 L405 355" class="blue"/>
<text x="405" y="470" class="txt">Creates</text>
<text x="390" y="510" class="txt">DB Connection</text>

<path d="M270 300 C330 360 360 360 410 300" class="blue"/>
<polygon points="410,300 380,280 380,320" fill="#2348c0"/>

<!-- Session Maker -->
<text x="760" y="80" class="title">Session</text>
<text x="760" y="120" class="title">Maker</text>
<!-- monitor icon -->
<rect x="760" y="180" width="180" height="120" rx="10" class="blue"/>
<path d="M850 300 V350 M790 350 H910" class="blue"/>
<text x="795" y="250" class="txt">Session</text>

<path d="M630 260 C690 260 710 260 760 260" class="blue"/>
<polygon points="760,260 730,240 730,280" fill="#2348c0"/>
<text x="650" y="220" class="txt">factory</text>

<!-- Identity Map -->
<text x="1050" y="80" class="title">Identity Map</text>
<rect x="1050" y="150" width="250" height="400" class="blue"/>
<circle cx="1120" cy="240" r="25" class="blue"/>
<circle cx="1120" cy="340" r="25" class="blue"/>
<circle cx="1120" cy="440" r="25" class="blue"/>
<text x="1170" y="250" class="txt">Object 1</text>
<text x="1170" y="350" class="txt">Object 2</text>
<text x="1170" y="450" class="txt">Object 3</text>

<path d="M940 240 C990 240 1000 240 1050 240" class="blue"/>
<polygon points="1050,240 1020,220 1020,260" fill="#2348c0"/>
<text x="960" y="210" class="txt">tracks</text>

<!-- Object lifecycle -->
<text x="100" y="650" class="title">ORM Object Lifecycle</text>

<circle cx="250" cy="760" r="70" class="red"/>
<text x="170" y="765" class="txt">Transient</text>

<circle cx="600" cy="760" r="70" class="blue"/>
<text x="535" y="765" class="txt">Pending</text>

<circle cx="950" cy="760" r="70" class="blue"/>
<text x="900" y="765" class="txt">Persistent</text>

<circle cx="1250" cy="760" r="70" class="red"/>
<text x="1190" y="765" class="txt">Detached</text>

<path d="M320 760 H530" class="blue"/>
<polygon points="530,760 500,740 500,780" fill="#2348c0"/>
<text x="370" y="720" class="txt">session.add()</text>

<path d="M670 760 H880" class="blue"/>
<polygon points="880,760 850,740 850,780" fill="#2348c0"/>
<text x="710" y="720" class="txt">flush()</text>

<path d="M1020 760 H1180" class="red"/>
<polygon points="1180,760 1150,740 1150,780" fill="#f04b4b"/>
<text x="1050" y="720" class="txt">close()</text>

<!-- Commit -->
<path d="M850 650 C850 600 850 560 850 550" class="blue"/>
<text x="760" y="620" class="txt">commit()</text>

<!-- Unit of work -->
<text x="430" y="600" class="orange">Session = Unit of Work</text>
<text x="430" y="640" class="orange">manages transaction + identity map</text>

</svg>
'''

path = Path("sqlalchemy_session_lifecycle_updated.svg")
path.write_text(svg, encoding="utf-8")

str(path)
