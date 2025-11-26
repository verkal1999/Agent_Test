// ua_test_server_secure.cpp
#include <cstdio>
#include <open62541/plugin/log_stdout.h>
#include <open62541/server.h>
#include <open62541/server_config_default.h>
#include <open62541/plugin/securitypolicy_default.h>
#include <open62541/plugin/accesscontrol_default.h>
#include <open62541/nodeids.h>

/* --------- Datei-Loader (DER) --------- */
static UA_ByteString loadFile(const char *path) {
    UA_ByteString bs = UA_BYTESTRING_NULL;
    FILE *f = fopen(path, "rb");
    if(!f) return bs;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    if(sz <= 0) { fclose(f); return bs; }
    fseek(f, 0, SEEK_SET);
    bs.length = (size_t)sz;
    bs.data   = (UA_Byte*)UA_malloc(bs.length);
    if(!bs.data) { fclose(f); bs.length = 0; return UA_BYTESTRING_NULL; }
    size_t rd = fread(bs.data, 1, bs.length, f);
    fclose(f);
    if(rd != bs.length) { UA_ByteString_clear(&bs); return UA_BYTESTRING_NULL; }
    return bs;
}

/* --------- Globale NodeIds / Status --------- */
static UA_NodeId gTriggerD2Id        = UA_NODEID_NULL;
static UA_NodeId gDiagnoseFinishedId = UA_NODEID_NULL;
static UA_NodeId gTestSkill1ActiveId = UA_NODEID_NULL;
static UA_NodeId gTestSkill2ActiveId = UA_NODEID_NULL;
static UA_NodeId gLastSkillId        = UA_NODEID_NULL;
static UA_NodeId gAutomatikbetriebId = UA_NODEID_NULL;
static UA_NodeId gZ1Id               = UA_NODEID_NULL;

static UA_Boolean gDiagPending = UA_FALSE;

static UA_Boolean gAutomatik = UA_TRUE;
static UA_UInt64  gTimerTS2  = 0;

/* --------- Schreib-Helfer --------- */
static void writeBool(UA_Server* s, const UA_NodeId& nid, UA_Boolean v) {
    UA_Variant var; UA_Variant_init(&var);
    UA_Variant_setScalar(&var, &v, &UA_TYPES[UA_TYPES_BOOLEAN]);
    UA_Server_writeValue(s, nid, var);
}
static void writeInt32(UA_Server* s, const UA_NodeId& nid, UA_Int32 v) {
    UA_Variant var; UA_Variant_init(&var);
    UA_Variant_setScalar(&var, &v, &UA_TYPES[UA_TYPES_INT32]);
    UA_Server_writeValue(s, nid, var);
}
static void writeString(UA_Server* s, const UA_NodeId& nid, const char* txt) {
    UA_String str = UA_STRING_ALLOC(const_cast<char*>(txt));
    UA_Variant var; UA_Variant_init(&var);
    UA_Variant_setScalar(&var, &str, &UA_TYPES[UA_TYPES_STRING]);
    UA_Server_writeValue(s, nid, var);
    UA_String_clear(&str);
}

/* --------- State Logging --------- */
static const char* tf(UA_Boolean b) { return b ? "TRUE" : "FALSE"; }

static void logState(UA_Server* server) {
    UA_Boolean v_trigger = UA_FALSE, v_diagFin = UA_FALSE, v_ts1 = UA_FALSE, v_ts2 = UA_FALSE, v_auto = UA_FALSE;
    UA_Int32   v_z1 = 0;
    UA_String  v_last = UA_STRING_NULL;

    UA_Variant v; UA_Variant_init(&v);

    if(UA_Server_readValue(server, gTriggerD2Id, &v) == UA_STATUSCODE_GOOD && UA_Variant_isScalar(&v) && v.type == &UA_TYPES[UA_TYPES_BOOLEAN] && v.data)
        v_trigger = *(UA_Boolean*)v.data;
    UA_Variant_clear(&v);

    if(UA_Server_readValue(server, gDiagnoseFinishedId, &v) == UA_STATUSCODE_GOOD && UA_Variant_isScalar(&v) && v.type == &UA_TYPES[UA_TYPES_BOOLEAN] && v.data)
        v_diagFin = *(UA_Boolean*)v.data;
    UA_Variant_clear(&v);

    if(UA_Server_readValue(server, gTestSkill1ActiveId, &v) == UA_STATUSCODE_GOOD && UA_Variant_isScalar(&v) && v.type == &UA_TYPES[UA_TYPES_BOOLEAN] && v.data)
        v_ts1 = *(UA_Boolean*)v.data;
    UA_Variant_clear(&v);

    if(UA_Server_readValue(server, gTestSkill2ActiveId, &v) == UA_STATUSCODE_GOOD && UA_Variant_isScalar(&v) && v.type == &UA_TYPES[UA_TYPES_BOOLEAN] && v.data)
        v_ts2 = *(UA_Boolean*)v.data;
    UA_Variant_clear(&v);

    if(UA_Server_readValue(server, gAutomatikbetriebId, &v) == UA_STATUSCODE_GOOD && UA_Variant_isScalar(&v) && v.type == &UA_TYPES[UA_TYPES_BOOLEAN] && v.data)
        v_auto = *(UA_Boolean*)v.data;
    UA_Variant_clear(&v);

    if(UA_Server_readValue(server, gZ1Id, &v) == UA_STATUSCODE_GOOD && UA_Variant_isScalar(&v) && v.type == &UA_TYPES[UA_TYPES_INT32] && v.data)
        v_z1 = *(UA_Int32*)v.data;
    UA_Variant_clear(&v);

    if(UA_Server_readValue(server, gLastSkillId, &v) == UA_STATUSCODE_GOOD && UA_Variant_isScalar(&v) && v.type == &UA_TYPES[UA_TYPES_STRING] && v.data)
        v_last = *(UA_String*)v.data;

    UA_LOG_INFO(UA_Log_Stdout, UA_LOGCATEGORY_SERVER,
        "[State] Auto=%s (g=%s) DiagPending=%s | TriggerD2=%s DiagnoseFinished=%s | TS1=%s TS2=%s | z1=%d | LastSkill=\"%.*s\"",
        tf(v_auto), tf(gAutomatik), tf(gDiagPending), tf(v_trigger), tf(v_diagFin), tf(v_ts1), tf(v_ts2),
        (int)v_z1, (int)v_last.length, (const char*)v_last.data);

    UA_Variant_clear(&v); // clears v_last if set
}

/* --------- TriggerD2 (Puls + Diagnoseanforderung) --------- */
static void TriggerD2_setFalse(UA_Server *server, void *data) {
    UA_NodeId *nid = (UA_NodeId*)data;
    writeBool(server, *nid, UA_FALSE);
    UA_LOG_INFO(UA_Log_Stdout, UA_LOGCATEGORY_SERVER, "[Server] TriggerD2 -> FALSE");
}
static void onTriggerD2Write(UA_Server *server, const UA_NodeId* /*sessionId*/, void* /*sessionCtx*/,
                             const UA_NodeId* /*nodeId*/, void* /*nodeCtx*/, const UA_NumericRange* /*range*/,
                             const UA_DataValue *data) {
    if(!data || !data->hasValue) return;
    if(UA_Variant_isScalar(&data->value) &&
       data->value.type == &UA_TYPES[UA_TYPES_BOOLEAN] && data->value.data) {
        UA_Boolean b = *(UA_Boolean*)data->value.data;
        if(b == UA_TRUE) {
            /* Automatik stoppen und Diagnose markieren */
            writeBool(server, gAutomatikbetriebId, UA_FALSE);
            gDiagPending = UA_TRUE;
            /* 200 ms später Trigger zurücksetzen */
            UA_DateTime when = UA_DateTime_nowMonotonic() + (UA_DateTime)(200 * UA_DATETIME_MSEC);
            /* waiting for DiagnoseFinished -> no auto-reset here */
            UA_LOG_INFO(UA_Log_Stdout, UA_LOGCATEGORY_SERVER, "[Server] TriggerD2 -> TRUE");
            logState(server);
        }
    }
}


/* Forward declaration: TS2_start is used in TS1_start */
static void TS2_start(UA_Server*, void*);

/* --------- TriggerD2 periodic pulse --------- */
static void TriggerD2_pulse(UA_Server* server, void*) {
    // Nur auslösen, wenn Automatik aktiv und keine Diagnose läuft
    if(!gAutomatik || gDiagPending) return;

    gDiagPending = UA_TRUE;
    gAutomatik   = UA_FALSE;

    // Evtl. geplanten TS2-Start stoppen
    if(gTimerTS2) {
        UA_Server_removeCallback(server, gTimerTS2);
        gTimerTS2 = 0;
    }

    // Automatikbetrieb serverseitig sperren und Trigger setzen
    writeBool(server, gAutomatikbetriebId, UA_FALSE);
    writeBool(server, gTriggerD2Id, UA_TRUE);

    UA_LOG_INFO(UA_Log_Stdout, UA_LOGCATEGORY_SERVER, "[Server] TriggerD2 -> TRUE (auto)");
    // Kein Auto-Reset hier: Server wartet auf DiagnoseFinished=TRUE vom Client
}

static void TS1_stop(UA_Server* server, void*) { writeBool(server, gTestSkill1ActiveId, UA_FALSE); logState(server); }
static void TS2_stop(UA_Server* server, void*) { writeBool(server, gTestSkill2ActiveId, UA_FALSE); logState(server); }

static void TS1_start(UA_Server* server, void*) {
    if(!gAutomatik || gDiagPending) return;
    writeBool(server, gTestSkill1ActiveId, UA_TRUE);
    UA_LOG_INFO(UA_Log_Stdout, UA_LOGCATEGORY_SERVER, "[Server] TestSkill1 gestartet");
    writeString(server, gLastSkillId, "TestSkill1");
    UA_DateTime whenStop = UA_DateTime_nowMonotonic() + (UA_DateTime)(30000 * UA_DATETIME_MSEC); // 30 s
    UA_Server_addTimedCallback(server, TS1_stop, NULL, whenStop, NULL);
    /* TS2 in 30 s planen */
    UA_DateTime whenTS2 = UA_DateTime_nowMonotonic() + (UA_DateTime)(30000 * UA_DATETIME_MSEC);
    UA_Server_addTimedCallback(server, TS2_start, NULL, whenTS2, &gTimerTS2);
    logState(server);
}

static void TS2_start(UA_Server* server, void*) {
    if(!gAutomatik || gDiagPending) { gTimerTS2 = 0; return; }
    gTimerTS2 = 0;
    writeBool(server, gTestSkill2ActiveId, UA_TRUE);
    UA_LOG_INFO(UA_Log_Stdout, UA_LOGCATEGORY_SERVER, "[Server] TestSkill2 gestartet");
    writeString(server, gLastSkillId, "TestSkill2");
    UA_DateTime when = UA_DateTime_nowMonotonic() + (UA_DateTime)(10000 * UA_DATETIME_MSEC); // 10 s
    UA_Server_addTimedCallback(server, TS2_stop, NULL, when, NULL);
    logState(server);
}

/* --------- Einfacher Zähler z1 (1/s) --------- */
static void z1_inc(UA_Server* server, void*) {
    UA_Variant var; UA_Variant_init(&var);
    UA_Int32 cur = 0;
    if(UA_Server_readValue(server, gZ1Id, &var) == UA_STATUSCODE_GOOD &&
       UA_Variant_isScalar(&var) &&
       var.type == &UA_TYPES[UA_TYPES_INT32] && var.data) {
        cur = *static_cast<UA_Int32*>(var.data);
    }
    writeInt32(server, gZ1Id, cur + 1);
    logState(server);
}

/* --------- Write-Callback: DiagnoseFinished (void-Signatur!) --------- */
static void onDiagnoseFinishedWrite(UA_Server        *server,
                                    const UA_NodeId  * /*sessionId*/, void * /*sessionCtx*/,
                                    const UA_NodeId  * /*nodeId*/,   void * /*nodeCtx*/,
                                    const UA_NumericRange * /*range*/,
                                    const UA_DataValue    *data) {
    if(!data || !data->hasValue) return;
    if(UA_Variant_isScalar(&data->value) &&
       data->value.type == &UA_TYPES[UA_TYPES_BOOLEAN] &&
       data->value.data) {
        UA_Boolean b = *static_cast<UA_Boolean*>(data->value.data);
        if(b == UA_TRUE) {
            /* Client signalisiert: Diagnose beendet. -> TriggerD2 sofort FALSE, Automatik TRUE */
            UA_LOG_INFO(UA_Log_Stdout, UA_LOGCATEGORY_SERVER,
                        "[Server] DiagnoseFinished=TRUE -> TriggerD2=FALSE, Automatikbetrieb=TRUE");
            gDiagPending = UA_FALSE;
            writeBool(server, gTriggerD2Id, UA_FALSE);
            writeBool(server, gAutomatikbetriebId, UA_TRUE);
            writeBool(server, gDiagnoseFinishedId, UA_FALSE); // Quittieren
            logState(server);
        }
    }
}

/* --------- Write-Callback: Automatikbetrieb (Zustand pflegen, TS2-Timer ggf. abbrechen) --------- */
static void onAutomatikWrite(UA_Server *server,
                             const UA_NodeId* /*sessionId*/, void* /*sessionCtx*/,
                             const UA_NodeId* /*nodeId*/, void* /*nodeCtx*/,
                             const UA_NumericRange* /*range*/,
                             const UA_DataValue *data) {
    if(!data || !data->hasValue) return;
    if(UA_Variant_isScalar(&data->value) &&
       data->value.type == &UA_TYPES[UA_TYPES_BOOLEAN] &&
       data->value.data) {
        gAutomatik = *(UA_Boolean*)data->value.data;
        if(gAutomatik == UA_FALSE) {
            if(gTimerTS2) { UA_Server_removeCallback(server, gTimerTS2); gTimerTS2 = 0; }
        }
    }
}

/* --------- main --------- */
int main() {
    UA_StatusCode ret = UA_STATUSCODE_GOOD;

    /* Server und Default-Konfiguration */
    UA_Server *server = UA_Server_new();
    UA_ServerConfig *config = UA_Server_getConfig(server);
    UA_ServerConfig_setMinimal(config, 4850, NULL);

    /* Zertifikate laden (Beispielpfade anpassen!) */
    UA_ByteString cert = loadFile("certs/server_cert.der");
    UA_ByteString key  = loadFile("certs/server_key.der");
    if(cert.length == 0 || key.length == 0) {
        UA_LOG_WARNING(UA_Log_Stdout, UA_LOGCATEGORY_SERVER, "Keine Zertifikatsdateien gefunden");
    }

    /* SecurityPolicy + Endpoint */
    UA_ServerConfig_addSecurityPolicyBasic256Sha256(config, &cert, &key);
    UA_String policyUri = UA_STRING_ALLOC("http://opcfoundation.org/UA/SecurityPolicy#Basic256Sha256");
    UA_ServerConfig_addEndpoint(config, policyUri, UA_MESSAGESECURITYMODE_SIGNANDENCRYPT);

    /* Access Control: Anonymous AUS, Username/Passwort AN (user/pass) */
    static const UA_UsernamePasswordLogin users[1] = {
        { UA_STRING_STATIC("user"), UA_STRING_STATIC("pass") }
    };
    UA_AccessControl_default(config, UA_FALSE, &policyUri, 1, users);
    UA_String_clear(&policyUri);

    /* ns=1 Variablen anlegen */
    auto addBool = [&](const char* name, UA_Boolean init, UA_NodeId& outId) {
        UA_VariableAttributes a = UA_VariableAttributes_default;
        UA_Variant_setScalar(&a.value, &init, &UA_TYPES[UA_TYPES_BOOLEAN]);
        a.displayName = UA_LOCALIZEDTEXT("en-US", const_cast<char*>(name));
        a.dataType    = UA_TYPES[UA_TYPES_BOOLEAN].typeId;
        a.accessLevel = UA_ACCESSLEVELMASK_READ | UA_ACCESSLEVELMASK_WRITE;
        return UA_Server_addVariableNode(server,
            UA_NODEID_STRING(1, const_cast<char*>(name)),
            UA_NODEID_NUMERIC(0, UA_NS0ID_OBJECTSFOLDER),
            UA_NODEID_NUMERIC(0, UA_NS0ID_ORGANIZES),
            UA_QUALIFIEDNAME(1, const_cast<char*>(name)),
            UA_NODEID_NUMERIC(0, UA_NS0ID_BASEDATAVARIABLETYPE),
            a, NULL, &outId);
    };
    auto addInt = [&](const char* name, UA_Int32 init, UA_NodeId& outId) {
        UA_VariableAttributes a = UA_VariableAttributes_default;
        UA_Variant_setScalar(&a.value, &init, &UA_TYPES[UA_TYPES_INT32]);
        a.displayName = UA_LOCALIZEDTEXT("en-US", const_cast<char*>(name));
        a.dataType    = UA_TYPES[UA_TYPES_INT32].typeId;
        a.accessLevel = UA_ACCESSLEVELMASK_READ | UA_ACCESSLEVELMASK_WRITE;
        return UA_Server_addVariableNode(server,
            UA_NODEID_STRING(1, const_cast<char*>(name)),
            UA_NODEID_NUMERIC(0, UA_NS0ID_OBJECTSFOLDER),
            UA_NODEID_NUMERIC(0, UA_NS0ID_ORGANIZES),
            UA_QUALIFIEDNAME(1, const_cast<char*>(name)),
            UA_NODEID_NUMERIC(0, UA_NS0ID_BASEDATAVARIABLETYPE),
            a, NULL, &outId);
    };
    auto addStr = [&](const char* name, const char* init, UA_NodeId& outId) {
        UA_VariableAttributes a = UA_VariableAttributes_default;
        UA_String str = UA_STRING_ALLOC(const_cast<char*>(init));
        UA_Variant_setScalar(&a.value, &str, &UA_TYPES[UA_TYPES_STRING]);
        UA_String_clear(&str);
        a.displayName = UA_LOCALIZEDTEXT("en-US", const_cast<char*>(name));
        a.dataType    = UA_TYPES[UA_TYPES_STRING].typeId;
        a.accessLevel = UA_ACCESSLEVELMASK_READ | UA_ACCESSLEVELMASK_WRITE;
        return UA_Server_addVariableNode(server,
            UA_NODEID_STRING(1, const_cast<char*>(name)),
            UA_NODEID_NUMERIC(0, UA_NS0ID_OBJECTSFOLDER),
            UA_NODEID_NUMERIC(0, UA_NS0ID_ORGANIZES),
            UA_QUALIFIEDNAME(1, const_cast<char*>(name)),
            UA_NODEID_NUMERIC(0, UA_NS0ID_BASEDATAVARIABLETYPE),
            a, NULL, &outId);
    };

    addBool("TriggerD2",         UA_FALSE, gTriggerD2Id);
    addBool("DiagnoseFinished",  UA_FALSE, gDiagnoseFinishedId);
    addBool("TestSkill1_active", UA_FALSE, gTestSkill1ActiveId);
    addBool("TestSkill2_active", UA_FALSE, gTestSkill2ActiveId);
    addStr ("LastSkill",         "",       gLastSkillId);
    addBool("Automatikbetrieb",  UA_TRUE,  gAutomatikbetriebId);
    addInt ("z1",                0,        gZ1Id);

    /* Write-Callback auf DiagnoseFinished (void-Signatur in deiner Version) */
    {
        UA_ValueCallback cb; cb.onRead = nullptr; cb.onWrite = onDiagnoseFinishedWrite;
        UA_Server_setVariableNode_valueCallback(server, gDiagnoseFinishedId, cb);
    }
    /* Write-Callback auf TriggerD2 (ereignisgesteuert statt Repeater) */
    {
        UA_ValueCallback cb; cb.onRead = nullptr; cb.onWrite = onTriggerD2Write;
        UA_Server_setVariableNode_valueCallback(server, gTriggerD2Id, cb);
    }
    /* Write-Callback auf Automatikbetrieb (Zustand & Timerpflege) */
    {
        UA_ValueCallback cb; cb.onRead = nullptr; cb.onWrite = onAutomatikWrite;
        UA_Server_setVariableNode_valueCallback(server, gAutomatikbetriebId, cb);
    }

    logState(server);

    /* Periodische Callbacks */
    UA_UInt64 repTS1 = 0, repZ1 = 0, repTrig = 0;
    UA_Server_addRepeatedCallback(server, TriggerD2_pulse, nullptr,        60000.0, &repTrig);
    UA_Server_addRepeatedCallback(server, TS1_start,     nullptr,        60000.0, &repTS1);  // 60 s
    UA_Server_addRepeatedCallback(server, z1_inc,        nullptr,        1000.0,  &repZ1);   // 1 s

    UA_LOG_INFO(UA_Log_Stdout, UA_LOGCATEGORY_SERVER,
        "[Server] Secure UA Server laeuft auf opc.tcp://localhost:4850 (Basic256Sha256, Sign&Encrypt, User/Pass)");

    UA_Boolean running = true;
    ret = UA_Server_run(server, &running);

    UA_Server_delete(server);
    UA_ByteString_clear(&cert);
    UA_ByteString_clear(&key);
    return (ret == UA_STATUSCODE_GOOD) ? 0 : (int)ret;
}
