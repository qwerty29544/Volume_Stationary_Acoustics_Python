// Gmsh project created on Sat Mar 11 09:43:34 2023
//+
Point(1) = {-1, -1, 0, 1.0};
//+
Point(2) = {-1, -0, 0, 1.0};
//+
Point(3) = {-0, 0, 0, 1.0};
//+
Point(4) = {-0, -1, 0, 1.0};
//+
Line(1) = {2, 3};
//+
Line(2) = {3, 4};
//+
Line(3) = {4, 1};
//+
Line(4) = {2, 1};
//+
Point(5) = {-0.25, -0.75, 0, 1.0};
//+
Point(6) = {-0.75, -0.75, 0, 1.0};
//+
Point(7) = {-0.75, -0.25, 0, 1.0};
//+
Point(8) = {-0.25, -0.25, 0, 1.0};
//+
Line(5) = {7, 6};
//+
Line(6) = {6, 5};
//+
Line(7) = {5, 8};
//+
Line(8) = {8, 7};
//+
SetFactory("OpenCASCADE");
//+
SetFactory("OpenCASCADE");
//+
SetFactory("OpenCASCADE");
//+
Point(9) = {-0.25, -1, 0, 1.0};
//+
Point(10) = {-0.5, -1, 0, 1.0};
//+
Point(11) = {-0.75, -1, 0, 1.0};
//+
Point(12) = {-1, -0.75, 0, 1.0};
//+
Point(13) = {-1, -0.5, 0, 1.0};
//+
Point(14) = {-1, -0.25, 0, 1.0};
//+
Point(15) = {-0.75, 0, 0, 1.0};
//+
Point(16) = {-0.5, 0, 0, 1.0};
//+
Point(17) = {-0.25, 0, 0, 1.0};
//+
Point(18) = {0, -0.25, 0, 1.0};
//+
Point(19) = {0, -0.5, 0, 1.0};
//+
Point(20) = {0, -0.75, 0, 1.0};
//+
Point(21) = {-0.5, -0.75, 0, 1.0};
//+
Point(22) = {-0.75, -0.5, 0, 1.0};
//+
Point(23) = {-0.5, -0.25, 0, 1.0};
//+
Point(24) = {-0.25, -0.5, 0, 1.0};