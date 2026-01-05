import React, { useEffect, useRef, useState } from 'react';
// CHANGE 1: Import 'CandlestickSeries' explicitly
import { createChart, ColorType, CandlestickSeries } from 'lightweight-charts';
import axios from 'axios';

const App = () => {
    const chartContainerRef = useRef();
    const [botStatus, setBotStatus] = useState({ active: false, equity: 10000, decision: "WAIT", confidence: 0.0 });
    const [timeframe, setTimeframe] = useState("1h");

    // Chart References
    const candlestickSeriesRef = useRef();

    useEffect(() => {
        // 1. Create Chart
        const chart = createChart(chartContainerRef.current, {
            layout: { background: { type: ColorType.Solid, color: '#1E1E1E' }, textColor: '#DDD' },
            grid: { vertLines: { color: '#2B2B43' }, horzLines: { color: '#2B2B43' } },
            width: chartContainerRef.current.clientWidth,
            height: 500,
        });

        // CHANGE 2: Use addSeries(CandlestickSeries, options) instead of addCandlestickSeries(options)
        const newSeries = chart.addSeries(CandlestickSeries, {
            upColor: '#26a69a', 
            downColor: '#ef5350', 
            borderVisible: false,
            wickUpColor: '#26a69a', 
            wickDownColor: '#ef5350',
        });
        
        candlestickSeriesRef.current = newSeries;

        // Fetch History
        axios.get(`http://localhost:8000/history/${timeframe}`)
            .then(res => {
                // Ensure data is sorted by time to prevent library errors
                const sortedData = res.data.sort((a, b) => a.time - b.time);
                newSeries.setData(sortedData);
            })
            .catch(err => console.error("History Fetch Error:", err));

        // 2. WebSocket
        const ws = new WebSocket(`ws://localhost:8000/ws/${timeframe}`);
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            newSeries.update(data.candle);
            setBotStatus({
                active: data.bot.is_active,
                decision: data.bot.decision,
                confidence: data.bot.confidence,
                equity: data.bot.equity
            });
        };

        // Resize handler
        const handleResize = () => {
            chart.applyOptions({ width: chartContainerRef.current.clientWidth });
        };
        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            chart.remove();
            ws.close();
        };
    }, [timeframe]);

    const toggleBot = async () => {
        if (botStatus.active) await axios.post("http://localhost:8000/bot/stop");
        else await axios.post("http://localhost:8000/bot/start");
    };

    return (
        <div style={{ backgroundColor: "#121212", color: "white", minHeight: "100vh", padding: "20px", fontFamily: "sans-serif" }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "20px" }}>
                <h1>🤖 AI Trading Bot</h1>
                <div style={{ textAlign: "right" }}>
                    <div style={{ fontSize: "24px", marginBottom: "10px" }}>
                        Equity: <b style={{ color: "#4caf50" }}>${botStatus.equity?.toFixed(2)}</b>
                    </div>
                    <button 
                        onClick={toggleBot}
                        style={{
                            padding: "10px 20px", 
                            fontSize: "16px", 
                            fontWeight: "bold",
                            backgroundColor: botStatus.active ? "#ef5350" : "#26a69a",
                            color: "white", border: "none", cursor: "pointer", borderRadius: "5px"
                        }}
                    >
                        {botStatus.active ? "STOP BOT" : "START BOT"}
                    </button>
                </div>
            </div>

            <div style={{ marginBottom: "10px" }}>
                {['15m', '1h', '4h', '1d'].map(tf => (
                    <button 
                        key={tf} 
                        onClick={() => setTimeframe(tf)}
                        style={{ 
                            marginRight: "10px", 
                            padding: "8px 16px", 
                            backgroundColor: timeframe === tf ? "#2962ff" : "#333", 
                            color: "white", border: "none", cursor: "pointer", borderRadius: "4px" 
                        }}
                    >
                        {tf}
                    </button>
                ))}
            </div>

            <div ref={chartContainerRef} style={{ border: "1px solid #333", borderRadius: "8px", overflow: "hidden" }} />

            <div style={{ marginTop: "20px", padding: "20px", backgroundColor: "#1E1E1E", border: "1px solid #333", borderRadius: "8px", display: "flex", gap: "50px" }}>
                <div>
                    <h3>Last Signal</h3>
                    <div style={{ fontSize: "2rem", color: botStatus.decision === "BUY" ? "#26a69a" : "#aaa" }}>
                        {botStatus.decision}
                    </div>
                </div>
                <div>
                    <h3>AI Confidence</h3>
                    <div style={{ fontSize: "2rem" }}>
                        {(botStatus.confidence * 100).toFixed(1)}%
                    </div>
                </div>
            </div>
        </div>
    );
};

export default App;