========================================================================
    CONSOLE APPLICATION : ray Project Overview
========================================================================


Project 3 - Ray Tracing
Kyle Berry
scene file: test.scene
Grayscaling is selected by changing the global boolean variable, gray, in ray.cpp
Due to unknown glitch, obj models are placed farther back than they should. This
can be seen with grayscale. In test.scene dolphins.obj are placed at 0 0.2 0.

The scene file is rendered at 501 x 501 because 500 x 500 wasn't drawing
for some reason.

Implements reflection but not refraction.


The method for detecting shadows cast by spheres, shadow_sphere_intersect,
doesn't work and has been commented out as it caused spheres to cast shadows
on themselves at inappropriate angles. Uncomment to see a lot of black.






AppWizard has created this ray application for you.  

This file contains a summary of what you will find in each of the files that
make up your ray application.


ray.vcproj
    This is the main project file for VC++ projects generated using an Application Wizard. 
    It contains information about the version of Visual C++ that generated the file, and 
    information about the platforms, configurations, and project features selected with the
    Application Wizard.

ray.cpp
    This is the main application source file.

/////////////////////////////////////////////////////////////////////////////
Other standard files:

StdAfx.h, StdAfx.cpp
    These files are used to build a precompiled header (PCH) file
    named ray.pch and a precompiled types file named StdAfx.obj.

/////////////////////////////////////////////////////////////////////////////
Other notes:

AppWizard uses "TODO:" comments to indicate parts of the source code you
should add to or customize.

/////////////////////////////////////////////////////////////////////////////
