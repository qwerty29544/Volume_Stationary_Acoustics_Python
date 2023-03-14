// Gmsh project created on Sat Mar 11 11:26:49 2023
//+
SetFactory("OpenCASCADE");
Rectangle(1) = {0, 0, 0, 1, 1, 0};
//+
Circle(5) = {0.4, 0.6, -0, 0.25, 0, 2*Pi};
//+
Curve Loop(2) = {1, 2, 3, 4};
//+
Curve Loop(3) = {5};
//+
Surface(2) = {2, 3};
//+
Curve Loop(4) = {5};
//+
Surface(2) = {4};
//+
Curve Loop(6) = {3, 4, 1, 2};
