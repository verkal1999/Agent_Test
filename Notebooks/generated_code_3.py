import time

def FB_Betriebsarten(V_10000000001: bool, D2: bool, A5: bool, GVL__DOT__Start: bool, A1: bool, D1: bool, D3: bool, V_30000000002: bool, V_30000000005: bool, GVL__DOT__Weiter: bool, GVL__DOT__NotStopp, A6, V_30000000011: bool, Auto_Stoerung, F1, GVL__DOT__DiagnoseRequested, Alt_abort, V_40000000012: bool, Alt_found, D3_Requested, A2, Init_reached, Cycle_ended, A2_Requested, Start_Init):
    """Auto-generated from PLCopen XML (vereinfachte Semantik)."""

    def OR_1(V_10000000002, V_10000000003):
        result = D2 or A5
        return result
    V_10000000004 = OR_1(D2, A5)

    def RS_1(V_10000000001, V_10000000004):
        result = V_10000000004
        return result
    V_10000000005 = RS_1(V_10000000001, V_10000000004)

    def AND_1(V_20000000000, V_20000000001):
        result = GVL__DOT__Start and A1
        return result
    V_20000000002 = AND_1(GVL__DOT__Start, A1)

    def OR_2(V_20000000003, V_20000000004, V_20000000005):
        result = D2 or D1 or D3
        return result
    V_20000000006 = OR_2(D2, D1, D3)

    def RS_2(V_20000000002, V_20000000006):
        result = V_20000000006
        return result
    V_20000000007 = RS_2(V_20000000002, V_20000000006)

    def AND_2(V_30000000001, V_30000000002):
        result = D1 and V_30000000002
        return result
    V_30000000003 = AND_2(D1, V_30000000002)

    def OR_3(V_30000000000, V_30000000003):
        result = D2 or V_30000000003
        return result
    V_30000000004 = OR_3(D2, V_30000000003)

    def AND_3(V_30000000004, V_30000000005, V_30000000006, V_30000000007):
        result = V_30000000004 and V_30000000005 and GVL__DOT__Weiter and GVL__DOT__NotStopp
        return result
    V_30000000008 = AND_3(V_30000000004, V_30000000005, GVL__DOT__Weiter, GVL__DOT__NotStopp)

    def AND_4(V_30000000010, V_30000000011):
        result = D1 and V_30000000011
        return result
    V_30000000012 = AND_4(D1, V_30000000011)

    def OR_4(V_30000000009, V_30000000012):
        result = A6 or V_30000000012
        return result
    V_30000000013 = OR_4(A6, V_30000000012)

    def RS_3(V_30000000008, V_30000000013):
        result = V_30000000013
        return result
    V_30000000014 = RS_3(V_30000000008, V_30000000013)

    def AND_5(V_40000000000, V_40000000001):
        result = Auto_Stoerung and F1
        return result
    V_40000000002 = AND_5(Auto_Stoerung, F1)

    def AND_6(V_40000000003, V_40000000004, V_40000000005):
        result = GVL__DOT__DiagnoseRequested and D1 and GVL__DOT__NotStopp
        return result
    V_40000000006 = AND_6(GVL__DOT__DiagnoseRequested, D1, GVL__DOT__NotStopp)

    def AND_7(V_40000000007, V_40000000008):
        result = D3 and Alt_abort
        return result
    V_40000000009 = AND_7(D3, Alt_abort)

    def OR_5(V_40000000002, V_40000000006, V_40000000009):
        result = V_40000000002 or V_40000000006 or V_40000000009
        return result
    V_40000000010 = OR_5(V_40000000002, V_40000000006, V_40000000009)

    def AND_8(V_40000000011, V_40000000012):
        result = D1 and V_40000000012
        return result
    V_40000000013 = AND_8(D1, V_40000000012)

    def OR_6(V_40000000013, V_40000000014, V_40000000015):
        result = V_40000000013 or D3 or A5
        return result
    V_40000000016 = OR_6(V_40000000013, D3, A5)

    def RS_4(V_40000000010, V_40000000016):
        result = V_40000000016
        return result
    V_40000000017 = RS_4(V_40000000010, V_40000000016)

    def AND_9(V_50000000000, V_50000000001):
        result = D2 and Alt_found
        return result
    V_50000000002 = AND_9(D2, Alt_found)

    def AND_10(V_50000000003, V_50000000004):
        result = F1 and D3_Requested
        return result
    V_50000000005 = AND_10(F1, D3_Requested)

    def OR_7(V_50000000002, V_50000000005):
        result = V_50000000002 or V_50000000005
        return result
    V_50000000006 = OR_7(V_50000000002, V_50000000005)

    def OR_8(V_50000000007, V_50000000008, V_50000000009):
        result = A2 or D1 or D2
        return result
    V_50000000010 = OR_8(A2, D1, D2)

    def RS_5(V_50000000006, V_50000000010):
        result = V_50000000010
        return result
    V_50000000011 = RS_5(V_50000000006, V_50000000010)

    def AND_11(V_60000000000, V_60000000001):
        result = A6 and Init_reached
        return result
    V_60000000002 = AND_11(A6, Init_reached)

    def AND_12(V_60000000003, V_60000000004, V_60000000005):
        result = A2 and Init_reached and Cycle_ended
        return result
    V_60000000006 = AND_12(A2, Init_reached, Cycle_ended)

    def OR_9(V_60000000002, V_60000000006):
        result = V_60000000002 or V_60000000006
        return result
    V_60000000007 = OR_9(V_60000000002, V_60000000006)

    def OR_10(V_60000000008, V_60000000009):
        result = F1 or D1
        return result
    V_60000000010 = OR_10(F1, D1)

    def RS_6(V_60000000007, V_60000000010):
        result = V_60000000010
        return result
    V_60000000011 = RS_6(V_60000000007, V_60000000010)

    def AND_13(V_70000000000, V_70000000001):
        result = D3 and A2_Requested
        return result
    V_70000000002 = AND_13(D3, A2_Requested)

    def OR_11(V_70000000003, V_70000000004):
        result = A1 or D1
        return result
    V_70000000005 = OR_11(A1, D1)

    def RS_7(V_70000000002, V_70000000005):
        result = V_70000000005
        return result
    V_70000000006 = RS_7(V_70000000002, V_70000000005)

    def R_TRIG(V_80000000000):
        result = Start_Init
        return result
    V_80000000001 = R_TRIG(Start_Init)

    def OR_12(V_80000000001, V_80000000002):
        result = V_80000000001 or A5
        return result
    V_80000000003 = OR_12(V_80000000001, A5)

    def OR_13(V_80000000004, V_80000000005):
        result = A1 or D1
        return result
    V_80000000006 = OR_13(A1, D1)

    def RS_8(V_80000000003, V_80000000006):
        result = V_80000000006
        return result
    V_80000000007 = RS_8(V_80000000003, V_80000000006)
    D1 = V_10000000005
    print('Value of D1:', D1)
    F1 = V_20000000007
    print('Value of F1:', F1)
    A5 = V_30000000014
    print('Value of A5:', A5)
    D2 = V_40000000017
    print('Value of D2:', D2)
    D3 = V_50000000011
    print('Value of D3:', D3)
    A1 = V_60000000011
    print('Value of A1:', A1)
    A2 = V_70000000006
    print('Value of A2:', A2)
    A6 = V_80000000007
    print('Value of A6:', A6)
    return {'D1': D1, 'F1': F1, 'A5': A5, 'D2': D2, 'D3': D3, 'A1': A1, 'A2': A2, 'A6': A6}