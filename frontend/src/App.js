import React, { useEffect, useRef, useState } from 'react';
import { createChart, ColorType, CandlestickSeries } from 'lightweight-charts';
import axios from 'axios';

// --- IMPORTS ---
import { getBotConfig, updateBotConfig } from './api';
import History from './History';

const App = () => {
    const chartContainerRef = useRef();
    const candlestickSeriesRef = useRef();
    
    // --- VIEW STATE (Tabs) ---
    const [view, setView] = useState("dashboard"); // 'dashboard' or 'history'

    // --- DATA STATE ---
    const [botStatus, setBotStatus] = useState({ 
        active: false, 
        equity: 10000, 
        decision: "WAIT", 
        confidence: 0.0,
        pnl: 0.0,
        current_pos: null, 
        btc_balance: 0.0,
        mode: "FUTURES" 
    });

    // --- CONFIG STATE (Steering Wheel) ---
    const [config, setConfig] = useState({
        selected_timeframe: "1h",
        selected_model: "xgb",
        trend_bias: "NEUTRAL"
    });

    const [currentPrice, setCurrentPrice] = useState(0);

    // --- 1. INITIAL SETUP ---
    useEffect(() => {
        getBotConfig().then(data => {
            if (data) {
                setConfig(prev => ({
                    ...prev,
                    selected_timeframe: data.selected_timeframe,
                    selected_model: data.selected_model,
                    trend_bias: data.trend_bias
                }));
            }
        });
    }, []);

    // --- 2. CHART & WEBSOCKET ---
    useEffect(() => {
        // Only run chart logic if we are in Dashboard view
        if (view !== "dashboard") return;

        let isMounted = true;
        
        // Init Chart
        const chart = createChart(chartContainerRef.current, {
            layout: { background: { type: ColorType.Solid, color: '#121212' }, textColor: '#DDD' },
            grid: { vertLines: { color: '#1f1f1f' }, horzLines: { color: '#1f1f1f' } },
            width: chartContainerRef.current.clientWidth,
            height: 500,
            timeScale: { timeVisible: true, secondsVisible: false },
        });

        const newSeries = chart.addSeries(CandlestickSeries, {
            upColor: '#26a69a', downColor: '#ef5350', 
            borderVisible: false, wickUpColor: '#26a69a', wickDownColor: '#ef5350',
        });
        candlestickSeriesRef.current = newSeries;

        const handleResize = () => {
            if (chartContainerRef.current) {
                chart.applyOptions({ width: chartContainerRef.current.clientWidth });
            }
        };
        window.addEventListener('resize', handleResize);

        // Fetch History
        axios.get(`http://localhost:8000/history/${config.selected_timeframe}`)
            .then(res => {
                if (!isMounted) return;
                const sortedData = res.data.sort((a, b) => a.time - b.time);
                newSeries.setData(sortedData);
                if(sortedData.length > 0) setCurrentPrice(sortedData[sortedData.length - 1].close);
            })
            .catch(console.error);

        // WebSocket
        const ws = new WebSocket(`ws://localhost:8000/ws/${config.selected_timeframe}`);
        ws.onmessage = (event) => {
            if (!isMounted) return;
            const data = JSON.parse(event.data);
            
            if (data.candle) {
                newSeries.update(data.candle);
                setCurrentPrice(data.candle.close);
            }

            if (data.bot) {
                const detectedMode = data.bot.btc_balance !== undefined ? "SPOT" : "FUTURES";
                setBotStatus({
                    active: data.bot.is_active,
                    equity: data.bot.equity,
                    decision: data.bot.decision,
                    confidence: data.bot.confidence,
                    pnl: data.bot.pnl || 0.0,
                    current_pos: data.bot.positions,
                    btc_balance: data.bot.btc_balance || 0.0,
                    mode: detectedMode
                });
            }
        };

        return () => {
            isMounted = false;
            window.removeEventListener('resize', handleResize);
            chart.remove();
            ws.close();
        };
    }, [config.selected_timeframe, view]); // Re-run if Timeframe OR View changes

    // --- 3. CONTROLLERS ---
    const handleTimeframeChange = async (tf) => {
        setConfig(prev => ({ ...prev, selected_timeframe: tf }));
        await updateBotConfig({ selected_timeframe: tf });
    };

    const handleModelChange = async (e) => {
        const model = e.target.value;
        setConfig(prev => ({ ...prev, selected_model: model }));
        await updateBotConfig({ selected_model: model });
    };

    const handleTrendBiasChange = async (e) => {
        const bias = e.target.value;
        setConfig(prev => ({ ...prev, trend_bias: bias }));
        await updateBotConfig({ trend_bias: bias });
    };

    const toggleBot = async () => {
        const endpoint = botStatus.active ? "stop" : "start";
        setBotStatus(prev => ({ ...prev, active: !prev.active }));
        try {
            await axios.post(`http://localhost:8000/bot/${endpoint}`);
        } catch (err) {
            console.error(err);
            setBotStatus(prev => ({ ...prev, active: !prev.active }));
        }
    };

    const getSignalColor = (signal) => {
        if (signal?.includes("LONG") || signal?.includes("BUY")) return "#26a69a";
        if (signal?.includes("SHORT") || signal?.includes("SELL")) return "#ef5350";
        return "#aaa";
    };

    // --- 4. RENDER ---
    return (
        <div style={{ backgroundColor: "#000", color: "white", minHeight: "100vh", padding: "20px", fontFamily: "Inter, sans-serif" }}>
            
            {/* --- GLOBAL HEADER (Visible on both pages) --- */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "20px" }}>
                <div>
                    <h1 style={{ margin: 0 }}>
                        {botStatus.mode === "FUTURES" ? "🤖 Futures AI" : "🤖 Spot Accumulator"}
                    </h1>
                    <span style={{ fontSize: "12px", color: "#666", textTransform: "uppercase", letterSpacing: "1px" }}>
                        BTC/USDT • {botStatus.mode} • {config.selected_model.toUpperCase()}
                    </span>
                </div>
                
                <div style={{ textAlign: "right", display: "flex", gap: "20px", alignItems: "center" }}>
                    <div style={{ background: "#1E1E1E", padding: "10px 20px", borderRadius: "8px", border: "1px solid #333" }}>
                        <div style={{ fontSize: "12px", color: "#888" }}>Wallet Equity</div>
                        <div style={{ fontSize: "24px", fontWeight: "bold", color: "#fff" }}>
                            ${botStatus.equity?.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                        </div>
                    </div>

                    <button 
                        onClick={toggleBot}
                        style={{
                            padding: "12px 24px", fontSize: "16px", fontWeight: "bold",
                            backgroundColor: botStatus.active ? "#ef5350" : "#2962ff",
                            color: "white", border: "none", cursor: "pointer", borderRadius: "8px",
                            boxShadow: botStatus.active ? "0 0 15px rgba(239, 83, 80, 0.4)" : "0 0 15px rgba(41, 98, 255, 0.4)"
                        }}
                    >
                        {botStatus.active ? "STOP ENGINE" : "START ENGINE"}
                    </button>
                </div>
            </div>

            {/* --- NAVIGATION TABS --- */}
            <div style={{ marginBottom: "20px", display: "flex", gap: "10px" }}>
                <button 
                    onClick={() => setView("dashboard")}
                    style={{ ...navButtonStyle, background: view === "dashboard" ? "#2962ff" : "#222", color: view === "dashboard" ? "#fff" : "#888" }}
                >
                    📊 Live Dashboard
                </button>
                <button 
                    onClick={() => setView("history")}
                    style={{ ...navButtonStyle, background: view === "history" ? "#2962ff" : "#222", color: view === "history" ? "#fff" : "#888" }}
                >
                    📜 Trade History
                </button>
            </div>

            {/* --- CONTENT SWITCHER --- */}
            {view === "dashboard" ? (
                // --- DASHBOARD VIEW ---
                <>
                    {/* CONTROL PANEL */}
                    <div style={{ background: "#161616", padding: "15px", borderRadius: "12px", marginBottom: "20px", border: "1px solid #333", display: "flex", gap: "20px", alignItems: "center" }}>
                        {/* Timeframes */}
                        <div style={{ display: "flex", gap: "10px" }}>
                            {['1m', '5m', '15m', '1h'].map(tf => (
                                <button 
                                    key={tf} onClick={() => handleTimeframeChange(tf)}
                                    style={{ 
                                        padding: "6px 16px", borderRadius: "20px", cursor: "pointer", border: "none", fontWeight: "600",
                                        backgroundColor: config.selected_timeframe === tf ? "#2962ff" : "#222", 
                                        color: config.selected_timeframe === tf ? "#fff" : "#888",
                                        transition: "all 0.2s"
                                    }}
                                >
                                    {tf}
                                </button>
                            ))}
                        </div>

                        <div style={{ width: "1px", height: "30px", background: "#333" }}></div>

                        {/* Model Select */}
                        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                            <span style={{ color: "#888", fontSize: "14px" }}>Model:</span>
                            <select 
                                value={config.selected_model} onChange={handleModelChange}
                                style={{ background: "#222", color: "white", border: "1px solid #444", padding: "5px 10px", borderRadius: "5px" }}
                            >
                                <option value="xgb">XGBoost (Speed)</option>
                                <option value="rf">Random Forest (Stable)</option>
                                <option value="lgbm">LightGBM (Balanced)</option>
                                <option value="ensemble">Ensemble (Best)</option>
                                <option value="lstm">Deep Learning (LSTM)</option>
                            </select>
                        </div>

                        <div style={{ width: "1px", height: "30px", background: "#333" }}></div>

                        {/* Bias Select */}
                        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                            <span style={{ color: "#888", fontSize: "14px" }}>Trend Filter:</span>
                            <select 
                                value={config.trend_bias} onChange={handleTrendBiasChange}
                                style={{ background: "#222", color: "white", border: "1px solid #444", padding: "5px 10px", borderRadius: "5px" }}
                            >
                                <option value="NEUTRAL">No Filter (Long & Short)</option>
                                <option value="LONG_ONLY">🐂 Bull Market (Long Only)</option>
                                <option value="SHORT_ONLY">🐻 Bear Market (Short Only)</option>
                            </select>
                        </div>
                    </div>

                    {/* CHART */}
                    <div ref={chartContainerRef} style={{ border: "1px solid #222", borderRadius: "12px", overflow: "hidden", height: "500px", boxShadow: "0 10px 30px rgba(0,0,0,0.5)" }} />

                    {/* STATS */}
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "20px", marginTop: "20px" }}>
                        {/* Signal Card */}
                        <div style={{ background: "#1E1E1E", padding: "20px", borderRadius: "12px", borderLeft: `5px solid ${getSignalColor(botStatus.decision)}` }}>
                            <div style={{ color: "#888", fontSize: "14px" }}>AI Signal</div>
                            <div style={{ fontSize: "28px", fontWeight: "bold", color: getSignalColor(botStatus.decision) }}>
                                {botStatus.decision}
                            </div>
                            <div style={{ fontSize: "12px", color: "#555", marginTop: "5px" }}>
                                Confidence: {(botStatus.confidence * 100).toFixed(1)}%
                            </div>
                        </div>

                        {/* Position Status */}
                        <div style={{ background: "#1E1E1E", padding: "20px", borderRadius: "12px" }}>
                            {botStatus.mode === "FUTURES" ? (
                                <>
                                    <div style={{ color: "#888", fontSize: "14px" }}>Active Contract</div>
                                    {botStatus.current_pos ? (
                                        <>
                                            <div style={{ fontSize: "24px", fontWeight: "bold", color: "white" }}>
                                                {botStatus.current_pos.side} <span style={{fontSize: "14px", background: "#333", padding: "2px 6px", borderRadius: "4px"}}>5x</span>
                                            </div>
                                            <div style={{ fontSize: "12px", color: "#aaa" }}>
                                                Entry: ${botStatus.current_pos.entry_price?.toLocaleString()}
                                            </div>
                                        </>
                                    ) : (
                                        <div style={{ fontSize: "24px", color: "#555" }}>FLAT</div>
                                    )}
                                </>
                            ) : (
                                <>
                                    <div style={{ color: "#888", fontSize: "14px" }}>Asset Holdings</div>
                                    <div style={{ fontSize: "24px", fontWeight: "bold", color: botStatus.btc_balance > 0 ? "#26a69a" : "#555" }}>
                                        {botStatus.btc_balance?.toFixed(5)} BTC
                                    </div>
                                    <div style={{ fontSize: "12px", color: "#aaa" }}>
                                        {botStatus.btc_balance > 0 ? "Accumulating" : "Waiting in USD"}
                                    </div>
                                </>
                            )}
                        </div>

                        {/* PnL Card */}
                        <div style={{ background: "#1E1E1E", padding: "20px", borderRadius: "12px" }}>
                            {botStatus.mode === "FUTURES" ? (
                                <>
                                    <div style={{ color: "#888", fontSize: "14px" }}>Unrealized PnL</div>
                                    <div style={{ fontSize: "28px", fontWeight: "bold", color: botStatus.pnl >= 0 ? "#26a69a" : "#ef5350" }}>
                                        {botStatus.pnl >= 0 ? "+" : ""}${botStatus.pnl?.toFixed(2)}
                                    </div>
                                </>
                            ) : (
                                <>
                                    <div style={{ color: "#888", fontSize: "14px" }}>BTC Value</div>
                                    <div style={{ fontSize: "28px", fontWeight: "bold", color: "white" }}>
                                        ${(botStatus.btc_balance * currentPrice).toFixed(2)}
                                    </div>
                                </>
                            )}
                        </div>
                    </div>
                </>
            ) : (
                // --- HISTORY VIEW ---
                <History />
            )}
        </div>
    );
};

export default App;

// Helper Style
const navButtonStyle = { padding: "10px 20px", borderRadius: "8px", border: "none", cursor: "pointer", fontWeight: "bold", fontSize: "14px" };