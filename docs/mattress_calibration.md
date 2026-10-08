# Mattress calibration against a measured load-deflection curve

Vlaović et al. (2024) pressed the loading pad of EN 1957 into a polyurethane foam mattress and recorded the load against the deflection.
This page fits one parameter of the mattress model, the stiffness scale s, to the measured loads at 10 to 50 mm and compares the loads that the fitted model predicts at 60 to 100 mm with the measured ones.
All numbers are computed with `elastra-indentation` (`src/elastra/indentation.py`, `conf/indentation.yaml`) and the configuration of this repository.

## Setup

* Measured curve (`data/load_deflection/vlaovic2024_pur_100_cycles.csv`; source and digitisation in `data/README.md`): the mattress "PUR" (190 x 90 x 19 cm, a core of high-resilience polyurethane foam) after the first 100 cycles of the rolling durability test, loading branch, at 10 mm steps from 10 to 100 mm.
* Loading pad (EN 1957:2012, clause 5.6): a rigid disc 355 mm in diameter whose face has a convex spherical curvature of 800 mm radius, with a 20 mm front edge radius.
  In the model it is a mocap body with a mesh of this shape.
  The standard's pad has a smooth surface, so its contact with the cells is frictionless; solref and solimp are those of the mattress contact.
* Bed: the mattress of `conf/mattress/default.yaml` (20 x 19 cells, 0.1 m pitch) with `k` and `b` of bed `a1` multiplied by s; `c`, `d` and `m` are those of `a1`.
  The beds `a<n>` have `k` and `b` equal to n times those of `a1`, so s places the measured mattress among them.
* Loading: the bed settles under its own weight for 2 s; the pad is placed with its apex on the bed top, moved down at 10 mm/s to each deflection of the curve and held there for 1 s.
  The deflection is the depth of the pad apex below the top of the unloaded bed, and the load is the vertical contact force of the cells on the pad at the end of the hold.
  At the end of every hold no cell moves faster than 1e-9 m/s, so the loads are static.
* Fit: s minimises the sum of the squared differences between the simulated and the measured loads at 10, 20, 30, 40 and 50 mm (the fitting deflections).
  The load is proportional to s up to the contact compliance and the joint range, which stops a cell at z = 0, m g / k above its unloaded position; s is found by rescaling the bed by the least-squares factor and simulating it again, until that factor is within 1e-4 of 1 (three runs).
  The loads at 60, 70, 80, 90 and 100 mm (the held-out deflections) are not used in the fit.
* Physics step 0.625 ms.
  The same fit with a quarter of that step gives the same s and the same loads to a relative difference of 2e-11.
* The runs used an AMD Ryzen 9 9950X3D.

Commands (any `out=` directory):

```bash
elastra-indentation
elastra-indentation 'pad.position_xy_m=[0.05,0.0]' out=outputs/indentation/cell_centre
elastra-indentation mattress=refined sim.physics_dt_s=0.00015625 out=outputs/indentation/refined
elastra-indentation mattress=refined sim.physics_dt_s=0.00015625 \
    'pad.position_xy_m=[0.025,0.025]' out=outputs/indentation/refined_cell_centre
```

## Result

With the pad axis at the surface centre (the middle of the edge shared by two cells), the fitted stiffness scale is s = 4.10: `k` = 431.3 N/m and `b` = 207.7 N/m, between beds `a4` and `a8`.

| deflection (mm) | measured load (N) | simulated load (N) | relative error | used for |
| ---: | ---: | ---: | ---: | --- |
| 10 | 29.0 | 42.1 | +45.0 % | fit |
| 20 | 111.5 | 116.2 | +4.2 % | fit |
| 30 | 236.3 | 213.7 | -9.6 % | fit |
| 40 | 331.0 | 318.7 | -3.7 % | fit |
| 50 | 405.9 | 423.9 | +4.4 % | fit |
| 60 | 485.5 | 529.2 | +9.0 % | held out |
| 70 | 572.5 | 634.4 | +10.8 % | held out |
| 80 | 667.5 | 739.7 | +10.8 % | held out |
| 90 | 773.4 | 845.0 | +9.3 % | held out |
| 100 | 897.4 | 950.3 | +5.9 % | held out |

The fitted bed predicts the held-out loads with relative errors from +5.9 % to +10.8 %; it over-predicts every one of them.

From 30 mm on, 16 cells are in contact with the pad and the simulated load rises linearly, by 10.5 N/mm.
The measured curve is not linear.
Its slope falls from 12.5 N/mm between 20 and 30 mm to 7.5 N/mm between 40 and 50 mm and rises again to 12.4 N/mm between 90 and 100 mm: the plateau and the densification that Vlaović et al. describe for this foam.
It also starts soft: 29 N at 10 mm, where the simulated load is 45 % higher.

### No nonlinear compression term

The mattress model has no nonlinear compression term: the restoring force of every cell is linear in its deflection, and a bed is the five numbers `k`, `b`, `c`, `d`, `m`.
A term that stiffens the cells at large deflection, as foam densification does, would raise the predicted loads at the held-out deflections, which the linear bed already over-predicts.
The fitting deflections show no stiffening (the measured slope falls between 20 and 50 mm), so they cannot determine such a term.
A term that softens the cells, fitted to that fall of the slope, would carry the plateau on to 100 mm, where the measured curve stiffens again.
The model therefore keeps the linear cell spring.

## Grid spacing and pad position

The fit was repeated with the pad axis on a cell centre and on the 0.05 m grid (`mattress=refined`, physics step 0.15625 ms; `conf/mattress/refined.yaml` keeps the continuum of the 0.1 m grid, so s has the same meaning on both grids).
Relative error of the simulated load:

| grid, pad axis | s | 10 mm | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 m, (0, 0), cell edge | 4.10 | +45.0 % | +4.2 % | -9.6 % | -3.7 % | +4.4 % | +9.0 % | +10.8 % | +10.8 % | +9.3 % | +5.9 % |
| 0.1 m, (0.05, 0), cell centre | 3.76 | +49.7 % | -7.6 % | -16.8 % | -4.6 % | +7.2 % | +14.3 % | +17.8 % | +18.9 % | +18.1 % | +15.1 % |
| 0.05 m, (0, 0), cell corner | 4.74 | +37.2 % | -8.5 % | -18.5 % | -5.7 % | +8.3 % | +16.9 % | +21.2 % | +22.9 % | +22.4 % | +19.5 % |
| 0.05 m, (0.025, 0.025), cell centre | 4.87 | +41.5 % | -3.8 % | -15.4 % | -4.6 % | +6.8 % | +13.5 % | +16.6 % | +19.5 % | +20.2 % | +18.3 % |

The deflections 10 to 50 mm are the fitting deflections, 60 to 100 mm the held-out deflections.

Over the four cases s lies between 3.76 and 4.87 and the held-out errors between +5.9 % and +22.9 %, all of them over-predictions.
The default configuration (the 0.1 m grid of the robot experiments, pad axis at the surface centre) gives the smallest held-out errors; with the pad axis on a cell centre, or on the finer grid, they are 13.5 % to 22.9 %.
The fit therefore depends on the grid and on where the pad sits on it, as the load response does ([validation.md](validation.md), section 3): the bed's characteristic length, 0.083 m, is shorter than the 0.1 m cell pitch.
