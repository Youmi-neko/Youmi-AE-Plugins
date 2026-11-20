// Optic Bokeh Kantan (simple edition), v2.3: the params, included by OpticBokeh.cpp when OB_KANTAN is
// defined (inside its anonymous namespace; uses its helpers AddFloat / AddPopup / AddPicker).
// 26 visible controls (+ 3 thumbnail pickers, 2 groups, 1 hidden version): 14 outside the Blur Map group. The links from these
// controls to the full effect's values are in kantan.cpp; docs\kantan_design.md explains them.

// popup items from a shared table, without "Custom" (Kantan popups are the whole state)
template <typename T, size_t N>
std::string KantanItems(const T (&table)[N])
{
    std::string s;
    for (size_t i = 0; i < N; ++i)
    {
        if (i)
            s += "|";
        s += obs::T(table[i].ui);
    }
    return s;
}

PF_Err ParamsSetup(PF_InData* in_data, PF_OutData* out_data)
{
    PF_ParamDef def;
    PF_Err err = PF_Err_NONE;
    const PF_ValueDisplayFlags PCT = PF_ValueDisplayFlag_PERCENT, NONE = PF_ValueDisplayFlag_NONE;
    static const std::string lensNames = KantanItems(lenscore::kLensPresets);
    static const std::string styleNames = KantanItems(lenscore::kAberrationStyles);
    static const std::string stockNames = KantanItems(lenscore::kGrainStocks);
    const kantan::Values d;  // new-instance values

    // Lens: a preset is the lens's wide-open character; Aperture / Character / Bokeh Highlights move
    // away from it. Not supervised: it writes nothing.
    if ((err = AddPopup(in_data, obs::T("Lens"), lenscore::kLensPresetCount, d.lens, lensNames.c_str(), KDISK_LENS)))
        return err;
    if ((err = AddPicker(in_data, obs::T("Preset Thumbnails"), obthumb::LENS, KDISK_LENS_PICKER, false)))
        return err;
    if ((err = AddPopup(in_data, obs::T("Quality"), 3, d.quality, obs::T("Fast (Alpha)|Standard|Best (Beta)"), KDISK_QUALITY)))
        return err;
    if ((err = AddPopup(in_data, obs::T("View"), 4, d.view, obs::T("Result|Blur Amount|Highlight Source|Highlights Only"), KDISK_VIEW)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Radius"), 0, kantan::kRadiusMax, 0, kantan::kRadiusSliderMax, d.radius, d.radius, 1, NONE,
                        KDISK_RADIUS)))
        return err;

    // Blur Map: the full effect's map controls (Depth Strength = Map Scale in %, Depth Curve = Map Gamma)
    // plus the v2.3 map features (ob_mapx.h), the same as in the full effect
    PF_ADD_TOPICX(obs::T("Blur Map"), 0, KDISK_MAP_GROUP_START);
    AEFX_CLR_STRUCT(def);
    PF_ADD_LAYER(obs::T("Map Layer"), PF_LayerDefault_NONE, KDISK_MAP_LAYER);
    if ((err = AddPopup(in_data, obs::T("Map Channel"), 5, d.mapChannel, obs::T("Luminance|Red|Green|Blue|Alpha"), KDISK_MAP_CHANNEL)))
        return err;
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX(obs::T("Invert Map"), "", false, 0, KDISK_MAP_INVERT);
    if ((err = AddFloat(in_data, obs::T("Focus"), 0, 100, 0, 100, d.focusPct, d.focusPct, 1, PCT, KDISK_FOCUS)))
        return err;
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX(obs::T("Focus at Point"), "", false, 0, KDISK_FOCUS_POINT_ON);
    AEFX_CLR_STRUCT(def);
    PF_ADD_POINT(obs::T("Focus Point"), 50, 50, 0, KDISK_FOCUS_POINT);
    if ((err = AddMapFeature(in_data, MF_RANGE, KDISK_FOCUS_RANGE)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Depth Strength"), 0, kantan::kDepthStrengthMax, 0, kantan::kDepthStrengthSliderMax,
                        d.depthStrengthPct, d.depthStrengthPct, 1, PCT, KDISK_MAP_SCALE)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Depth Curve"), 0.1, 10, 0.2, 5, d.depthCurve, d.depthCurve, 2, NONE, KDISK_MAP_GAMMA)))
        return err;
    if ((err = AddMapFeature(in_data, MF_FRONT, KDISK_FRONT_BLUR)) || (err = AddMapFeature(in_data, MF_BACK, KDISK_BACK_BLUR)) ||
        (err = AddMapFeature(in_data, MF_BLUR, KDISK_MAP_BLUR)))
        return err;
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(KDISK_MAP_GROUP_END);

    // the three linked controls
    if ((err = AddFloat(in_data, obs::T("Aperture"), 0, kantan::kApertureMax, 0, kantan::kApertureMax, d.aperturePct, d.aperturePct,
                        1, PCT, KDISK_APERTURE)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Character"), 0, kantan::kCharacterMax, 0, kantan::kCharacterMax, d.characterPct,
                        d.characterPct, 1, PCT, KDISK_CHARACTER)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Chromatic Aberration"), 0, kantan::kChromaMax, 0, kantan::kChromaMax, d.chromaPct, d.chromaPct,
                        1, PCT, KDISK_CHROMA)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Bokeh Highlights"), 0, kantan::kHighlightsMax, 0, kantan::kHighlightsMax, d.highlightsPct,
                        d.highlightsPct, 1, PCT, KDISK_HIGHLIGHTS)))
        return err;

    // Film Look: three amounts (each moves its size / threshold / rim with it) and two style pickers
    PF_ADD_TOPICX(obs::T("Film Look"), PF_ParamFlag_START_COLLAPSED, KDISK_FILM_GROUP_START);
    if ((err = AddFloat(in_data, obs::T("Halation"), 0, kantan::kHalationMax, 0, 100, d.halationPct, d.halationPct, 1, PCT,
                        KDISK_HALATION)))
        return err;
    if ((err = AddPopup(in_data, obs::T("Aberration Style"), lenscore::kAberrationStyleCount, d.laStyle, styleNames.c_str(),
                        KDISK_LA_STYLE)))
        return err;
    if ((err = AddPicker(in_data, obs::T("Style Thumbnails"), obthumb::STYLE, KDISK_STYLE_PICKER, true)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Luminance Aberration"), 0, kantan::kLaMax, 0, 100, d.laPct, d.laPct, 1, PCT, KDISK_LA_AMOUNT)))
        return err;
    if ((err = AddPopup(in_data, obs::T("Grain Stock"), lenscore::kGrainStockCount, d.grainStock, stockNames.c_str(),
                        KDISK_GRAIN_STOCK)))
        return err;
    if ((err = AddPicker(in_data, obs::T("Stock Thumbnails"), obthumb::STOCK, KDISK_STOCK_PICKER, true)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Grain"), 0, kantan::kGrainMax, 0, kantan::kGrainMax, d.grainPct, d.grainPct, 1, PCT,
                        KDISK_GRAIN_AMOUNT)))
        return err;
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(KDISK_FILM_GROUP_END);

    if ((err = AddPopup(in_data, obs::T("Edge Behavior"), 3, d.edgeMode, obs::T("Spread|Repeat Edge|Clip to Layer"), KDISK_EDGE_MODE)))
        return err;
    // v2.4, out of the way at the bottom
    if ((err = AddPopup(in_data, obs::T("Use GPU"), 2, d.useGpu, obs::T("Auto|Off (CPU only)"), KDISK_USE_GPU)))
        return err;

    // Hidden link version (kantan.h). v2.3 = 1. When the links are retuned, the new version becomes the
    // dephault and value stays 1 with PF_ParamFlag_USE_VALUE_FOR_OLD_PROJECTS: saved projects keep the
    // curves they were made with.
    {
        AEFX_CLR_STRUCT(def);
        def.param_type = PF_Param_SLIDER;
        PF_STRNNCPY(def.PF_DEF_NAME, obs::T("Kantan Version"), sizeof(def.PF_DEF_NAME));
        def.u.sd.valid_min = def.u.sd.slider_min = 1;
        def.u.sd.valid_max = def.u.sd.slider_max = 99;
        def.u.sd.value = 1;
        def.u.sd.dephault = kantan::kLinkVersionCurrent;
        def.flags = PF_ParamFlag_CANNOT_TIME_VARY;
        def.ui_flags = PF_PUI_INVISIBLE;
        def.uu.id = KDISK_VERSION;
        if ((err = PF_ADD_PARAM(in_data, -1, &def)) != PF_Err_NONE)
            return err;
    }

    {
        PF_CustomUIInfo ci;
        AEFX_CLR_STRUCT(ci);
        ci.events = PF_CustomEFlag_EFFECT;
        if (in_data->inter.register_ui && (err = PF_REGISTER_UI(in_data, &ci)) != PF_Err_NONE)
            return err;
    }
    out_data->num_params = OB_NUM_PARAMS;
    return PF_Err_NONE;
}

// Kantan supervises nothing (the popups write no other param) and has no Best-only group.
PF_Err UserChangedParam(PF_InData*, PF_OutData*, PF_ParamDef*[], const PF_UserChangedParamExtra*)
{
    return PF_Err_NONE;
}
PF_Err UpdateParamsUI(PF_InData*, PF_OutData*, PF_ParamDef*[])
{
    return PF_Err_NONE;
}
