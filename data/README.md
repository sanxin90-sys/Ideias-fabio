# Input data (not included)

The scripts read one Excel workbook, which is **not redistributed** in this repository:

    data/Dados Rebio Jaru 2006 a 2008.xlsx

It holds half-hourly records from the micrometeorological tower of the Jaru Biological Reserve
(Rebio Jaru, Rondônia, Brazil; 10°11′11.4″ S, 61°52′29.9″ W), an LBA-network tower of 61.5 m above
open ombrophilous forest (mean canopy height about 33 m). Requests for the data should go to the
corresponding author (see the manuscript's data availability statement).

## Expected layout

- File name exactly as above (it is set in `rebio.py`, constant `XLSX`).
- Sheet **`DADOS OK`**, 17,568 rows (366 days × 48 half hours), one header row.
- The sheet has one block of columns per year, side by side, with repeated header names.
  pandas renames repeated headers with suffixes `.1`, `.2`, …, and the code uses those names.
- Row k holds the mean of the k-th half hour of the year; the first row of each year is 00:30.
  The calendar columns `Mês`, `Dia`, `Hora` (e.g. `30`, `100`, `130` for 00:30, 01:00, 01:30)
  follow a leap-year layout in every block; the code rebuilds the true calendar from the row index.

Columns used (names as pandas reads them):

| Year | Year column | Net radiation R (W m⁻²) | Air temperature (°C) | Relative humidity (%) | Wind speed (m s⁻¹) | T height |
| --- | --- | --- | --- | --- | --- | --- |
| 2006 | `Ano.2` | `R_liq.2` | `T_ar_22m.1` | `UR_62m.2` | `VV_ane_62m.2` | 22 m |
| 2007 | `Ano.3` | `R_liq.3` | `T_ar_22m.2` | `UR_62m.3` | `VV_ane_62m.3` | 22 m |
| 2008 | `Ano.4` | `R_liq.4` | `T_ar_62m.1` | `UR_62m.4` | `VV_ane_62m.4` | 62 m |
| 2008 (22 m control) | `Ano.4` | `R_liq.4` | `T_ar_22m.3` | `UR_62m.4` | `VV_ane_62m.4` | 22 m |
| 2009 | `Ano.5` | `R_liq filtro2` | `T_ar_62m.2` | `UR_62m.5` | `VV_ane_62m.5` | 62 m |
| 2010 | `Ano.6` | `R_liq filtro2.1` | `T_ar_62m.3` | `UR_62m.6` | `VV_ane_62m.6` | 62 m |

Step 01 also reads `Mês`, `Dia`, `Hora` (to select the 2008 seasons; 2008 is a leap year, so the layout is correct there) and carries precipitation `PRP.3` along, although no model uses it.

## Quality notes (see Sec. II A of the manuscript)

- Missing values are empty cells. Gaps up to 2 h are interpolated for the 2008 identification and
  up to 1 h for validation; longer gaps are left out.
- 2006 and 2007 have air temperature only at 22 m, inside the canopy. 2007 has low coverage.
- 2009 temperature at 62 m is 2–4 K colder than 2008 and 2010 in every month (suspected sensor or
  calibration issue). 2009–2010 net radiation comes from a filtered column (`R_liq filtro2`) whose
  equivalence to the 2008 product was not confirmed. These years are supplementary.
- Relative humidity exceeds 100% at times in 2009–2010 (up to 110.7%); the code caps it at 100%
  only where humidity is used (secondary wet-season closure).
- The scripts stop if T leaves 5–45 °C or R leaves −200 to 1400 W m⁻².
