=====================
Overlay Plot Examples
=====================

First subset some gridded data that will be used for the examples::

  cis subset od550aer:aerocom.HadGEM3-A-GLOMAP.A2.CTRL.monthly.od550aer.2006.nc t=[2006-10-13] -o HadGEM_od550aer-subset

  cis subset rsutcs:aerocom.HadGEM3-A-GLOMAP.A2.CTRL.monthly.rsutcs.2006.nc t=[2006-10-13] -o HadGEM_rsutcs-subset


Contour over heatmap
====================

::

  cis plot od550aer:HadGEM_od550aer-subset.nc:type=heatmap rsutcs:HadGEM_rsutcs-subset.nc:type=contour,color=white,contlevels=[1,10,25,50,175] --width 20 --height 15 --cbarscale 0.5 -o overlay1.png



.. image:: img/overlay1.png
   :width: 900px

::

  cis plot od550aer:HadGEM_od550aer-subset.nc:type=heatmap,cmap=binary rsutcs:HadGEM_rsutcs-subset.nc:type=contour,cmap=jet,contlevels=[1,10,25,50,175] --xmin -180 --xmax 180 --width 20 --height 15 --cbarscale 0.5 -o overlay2.png


.. image:: img/overlay2.png
   :width: 900px

Filled contour with transparency on NASA Blue Marble
====================================================

::

  cis plot od550aer:HadGEM_od550aer-subset.nc:cmap=Reds,type=contourf,transparency=0.5,cmin=0.15 --xmin -180 --xmax 180 --width 20 --height 15 --cbarscale 0.5 --nasabluemarble


.. image:: img/overlay3.png
   :width: 900px

Scatter plus Filled Contour
===========================

::

  cis subset rsutcs:HadGEM_rsutcs-subset.nc x=[-180,-90],y=[0,90] -o HadGEM_rsutcs-subset2

  cis plot GGALT:RF04.20090114.192600_035100.PNI.nc:type=scatter rsutcs:HadGEM_rsutcs-subset2.nc:type=contourf,contlevels=[0,10,20,30,40,50,100],transparency=0.7,contlabel=true,contfontsize=18 --width 20 --height 15 --xaxis longitude --yaxis latitude --xmin -180 --xmax -90 --ymin 0 --ymax 90 --itemwidth 20 -o overlay4.png


.. image:: img/overlay4.png
   :width: 600px

::

  cis plot GGALT:RF04.20090114.192600_035100.PNI.nc:type=scatter rsutcs:HadGEM_rsutcs-subset2.nc:type=contourf,contlevels=[40,50,100],transparency=0.3,contlabel=true,contfontsize=18,cmap=Reds --width 20 --height 15 --xaxis longitude --yaxis latitude --xmin -180 --xmax -90 --ymin 0 --ymax 90 --itemwidth 20 --nasabluemarble -o overlay5.png


.. image:: img/overlay5.png
   :width: 600px

File Locations
==============

The gridded data files can be found at::

  /group_workspaces/jasmin/cis/AeroCom/A2/HadGEM3-A-GLOMAP.A2.CTRL/renamed

and the ungridded::

  /group_workspaces/jasmin/cis/jasmin_cis_repo_test_files

