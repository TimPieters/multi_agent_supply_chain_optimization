\* CapacitatedFacilityLocation *\
Minimize
OBJ: 150 Open_0 + 180 Open_1 + 160 Open_2 + 170 Open_3 + 155 Open_4
 + 10.3333333333 Serve_0_0 + 12.4 Serve_0_1 + 15.5 Serve_0_2
 + 20.6666666667 Serve_0_3 + 18.6 Serve_0_4 + 14.4666666667 Serve_1_0
 + 11.3666666667 Serve_1_1 + 13.4333333333 Serve_1_2 + 16.5333333333 Serve_1_3
 + 19.6333333333 Serve_1_4 + 13.4333333333 Serve_2_0 + 17.5666666667 Serve_2_1
 + 12.4 Serve_2_2 + 14.4666666667 Serve_2_3 + 15.5 Serve_2_4 + 12.4 Serve_3_0
 + 15.5 Serve_3_1 + 10.3333333333 Serve_3_2 + 18.6 Serve_3_3
 + 16.5333333333 Serve_3_4 + 11.3666666667 Serve_4_0 + 13.4333333333 Serve_4_1
 + 14.4666666667 Serve_4_2 + 15.5 Serve_4_3 + 17.5666666667 Serve_4_4
Subject To
Capacity_0: - 80 Open_0 + 26 Serve_0_0 + 32 Serve_1_0 + 39 Serve_2_0
 + 23 Serve_3_0 + 29 Serve_4_0 <= 0
Capacity_1: - 90 Open_1 + 26 Serve_0_1 + 32 Serve_1_1 + 39 Serve_2_1
 + 23 Serve_3_1 + 29 Serve_4_1 <= 0
Capacity_2: - 70 Open_2 + 26 Serve_0_2 + 32 Serve_1_2 + 39 Serve_2_2
 + 23 Serve_3_2 + 29 Serve_4_2 <= 0
Capacity_3: - 100 Open_3 + 26 Serve_0_3 + 32 Serve_1_3 + 39 Serve_2_3
 + 23 Serve_3_3 + 29 Serve_4_3 <= 0
Capacity_4: - 85 Open_4 + 26 Serve_0_4 + 32 Serve_1_4 + 39 Serve_2_4
 + 23 Serve_3_4 + 29 Serve_4_4 <= 0
Demand_0: Serve_0_0 + Serve_0_1 + Serve_0_2 + Serve_0_3 + Serve_0_4 >= 1
Demand_1: Serve_1_0 + Serve_1_1 + Serve_1_2 + Serve_1_3 + Serve_1_4 >= 1
Demand_2: Serve_2_0 + Serve_2_1 + Serve_2_2 + Serve_2_3 + Serve_2_4 >= 1
Demand_3: Serve_3_0 + Serve_3_1 + Serve_3_2 + Serve_3_3 + Serve_3_4 >= 1
Demand_4: Serve_4_0 + Serve_4_1 + Serve_4_2 + Serve_4_3 + Serve_4_4 >= 1
Tightening_0_0: - Open_0 + Serve_0_0 <= 0
Tightening_0_1: - Open_1 + Serve_0_1 <= 0
Tightening_0_2: - Open_2 + Serve_0_2 <= 0
Tightening_0_3: - Open_3 + Serve_0_3 <= 0
Tightening_0_4: - Open_4 + Serve_0_4 <= 0
Tightening_1_0: - Open_0 + Serve_1_0 <= 0
Tightening_1_1: - Open_1 + Serve_1_1 <= 0
Tightening_1_2: - Open_2 + Serve_1_2 <= 0
Tightening_1_3: - Open_3 + Serve_1_3 <= 0
Tightening_1_4: - Open_4 + Serve_1_4 <= 0
Tightening_2_0: - Open_0 + Serve_2_0 <= 0
Tightening_2_1: - Open_1 + Serve_2_1 <= 0
Tightening_2_2: - Open_2 + Serve_2_2 <= 0
Tightening_2_3: - Open_3 + Serve_2_3 <= 0
Tightening_2_4: - Open_4 + Serve_2_4 <= 0
Tightening_3_0: - Open_0 + Serve_3_0 <= 0
Tightening_3_1: - Open_1 + Serve_3_1 <= 0
Tightening_3_2: - Open_2 + Serve_3_2 <= 0
Tightening_3_3: - Open_3 + Serve_3_3 <= 0
Tightening_3_4: - Open_4 + Serve_3_4 <= 0
Tightening_4_0: - Open_0 + Serve_4_0 <= 0
Tightening_4_1: - Open_1 + Serve_4_1 <= 0
Tightening_4_2: - Open_2 + Serve_4_2 <= 0
Tightening_4_3: - Open_3 + Serve_4_3 <= 0
Tightening_4_4: - Open_4 + Serve_4_4 <= 0
TotalDemand: 80 Open_0 + 90 Open_1 + 70 Open_2 + 100 Open_3 + 85 Open_4 >= 149
Bounds
 Serve_0_0 <= 1
 Serve_0_1 <= 1
 Serve_0_2 <= 1
 Serve_0_3 <= 1
 Serve_0_4 <= 1
 Serve_1_0 <= 1
 Serve_1_1 <= 1
 Serve_1_2 <= 1
 Serve_1_3 <= 1
 Serve_1_4 <= 1
 Serve_2_0 <= 1
 Serve_2_1 <= 1
 Serve_2_2 <= 1
 Serve_2_3 <= 1
 Serve_2_4 <= 1
 Serve_3_0 <= 1
 Serve_3_1 <= 1
 Serve_3_2 <= 1
 Serve_3_3 <= 1
 Serve_3_4 <= 1
 Serve_4_0 <= 1
 Serve_4_1 <= 1
 Serve_4_2 <= 1
 Serve_4_3 <= 1
 Serve_4_4 <= 1
Binaries
Open_0
Open_1
Open_2
Open_3
Open_4
End
