import time

def VSG_AS_CompressorControl(AlwaysTrue: bool, MethodCall_CompressorControl: UDint, ESS: UDint, Destination: UDint, V_20000000003, V_20000000004, V_20000000005, EncoderStartPosition01: bool, EncoderStartPosition02: UDint, EncoderStartPosition03: UDint, EncoderStartPosition04: bool, EncoderStartPosition05: string, EncoderStartPosition06: UDint, EncoderTargetPosition01: UDint, EncoderTargetPosition02: UDint, EncoderTargetPosition03: UDint, EncoderTargetPosition04: UDint, EncoderTargetPosition05: UDint, EncoderTargetPosition06, V_20000000018, V_20000000019, TON_TimeCompressor, TON_TimeValve, AutomaticMode):
    """Auto-generated from PLCopen XML (vereinfachte Semantik)."""

    def EXECUTE(V_10000000001):
        """EXECUTE block – original ST-Code:
CASE state OF
	0:
		IF jobRunning THEN
			state := state + 1;
		END_IF
		
	1:
		IF ResetMethodCall THEN
			jobRunning := FALSE;
			jobFinished := TRUE;
			state := 0;
			MethodCall_CompressorControl := FALSE;
		END_IF
		
END_CASE"""
        result = AlwaysTrue
        return result

    def CompressorControl(V_20000000000, V_20000000001, V_20000000002, V_20000000003, V_20000000004, V_20000000005, V_20000000006, V_20000000007, V_20000000008, V_20000000009, V_20000000010, V_20000000011, V_20000000012, V_20000000013, V_20000000014, V_20000000015, V_20000000016, V_20000000017, V_20000000018, V_20000000019, V_20000000020, V_20000000021, V_20000000022):
        result = AutomaticMode
        return result
    V_20000000023 = CompressorControl(MethodCall_CompressorControl, ESS, Destination, V_20000000003, V_20000000004, V_20000000005, EncoderStartPosition01, EncoderStartPosition02, EncoderStartPosition03, EncoderStartPosition04, EncoderStartPosition05, EncoderStartPosition06, EncoderTargetPosition01, EncoderTargetPosition02, EncoderTargetPosition03, EncoderTargetPosition04, EncoderTargetPosition05, EncoderTargetPosition06, V_20000000018, V_20000000019, TON_TimeCompressor, TON_TimeValve, AutomaticMode)
    DO_CompressorVSG = V_20000000023
    print('Value of DO_CompressorVSG:', DO_CompressorVSG)
    return {'DO_CompressorVSG': DO_CompressorVSG}