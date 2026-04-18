import React, { useEffect, useState } from 'react';
import { getTradeHistory } from './api';

const History = () => {
    const [trades, setTrades] = useState([]);
    const [stats, setStats] = useState({ total_pnl: 0, win_rate: 0, total_trades: 0 });

    useEffect(() => {
        loadHistory();
    }, []);

    const loadHistory = async () => {
        const data = await getTradeHistory();
        setTrades(data);
        calculateStats(data);
    };

    const calculateStats = (data) => {
        if (data.length === 0) return;

        // Filter only CLOSED trades for PnL calc
        const closed = data.filter(t => t.status === "CLOSED");
        const totalPnl = closed.reduce((acc, curr) => acc + (curr.pnl || 0), 0);
        const wins = closed.filter(t => t.pnl > 0).length;
        const winRate = closed.length > 0 ? (wins / closed.length) * 100 : 0;

        setStats({
            total_pnl: totalPnl,
            win_rate: winRate,
            total_trades: data.length
        });
    };

    const formatDate = (isoString) => {
        return new Date(isoString).toLocaleString();
    };

    return (
        <div style={{ padding: "20px", color: "#ddd", fontFamily: "Inter, sans-serif" }}>
            
            {/* STATS HEADER */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "20px", marginBottom: "30px" }}>
                <div style={cardStyle}>
                    <div style={labelStyle}>Total PnL</div>
                    <div style={{ ...valueStyle, color: stats.total_pnl >= 0 ? "#26a69a" : "#ef5350" }}>
                        ${stats.total_pnl.toFixed(2)}
                    </div>
                </div>
                <div style={cardStyle}>
                    <div style={labelStyle}>Win Rate</div>
                    <div style={valueStyle}>{stats.win_rate.toFixed(1)}%</div>
                </div>
                <div style={cardStyle}>
                    <div style={labelStyle}>Total Trades</div>
                    <div style={valueStyle}>{stats.total_trades}</div>
                </div>
            </div>

            {/* DATA TABLE */}
            <div style={{ background: "#1E1E1E", borderRadius: "12px", overflow: "hidden", border: "1px solid #333" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", textAlign: "left" }}>
                    <thead>
                        <tr style={{ background: "#252525", color: "#888", fontSize: "12px", textTransform: "uppercase" }}>
                            <th style={thStyle}>Time</th>
                            <th style={thStyle}>Side</th>
                            <th style={thStyle}>Price</th>
                            <th style={thStyle}>Size</th>
                            <th style={thStyle}>PnL</th>
                            <th style={thStyle}>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {trades.map((t, i) => (
                            <tr key={t.id || i} style={{ borderBottom: "1px solid #333" }}>
                                <td style={tdStyle}>{formatDate(t.timestamp)}</td>
                                <td style={{ ...tdStyle, color: t.side === "LONG" ? "#26a69a" : "#ef5350", fontWeight: "bold" }}>
                                    {t.side}
                                </td>
                                <td style={tdStyle}>${t.entry_price.toFixed(2)}</td>
                                <td style={tdStyle}>{t.size.toFixed(4)}</td>
                                <td style={{ ...tdStyle, color: t.pnl > 0 ? "#26a69a" : (t.pnl < 0 ? "#ef5350" : "#ddd") }}>
                                    {t.pnl ? `$${t.pnl.toFixed(2)}` : "-"}
                                </td>
                                <td style={tdStyle}>
                                    <span style={{ 
                                        padding: "4px 8px", borderRadius: "4px", fontSize: "11px",
                                        background: t.status === "OPEN" ? "#2962ff" : "#444"
                                    }}>
                                        {t.status}
                                    </span>
                                </td>
                            </tr>
                        ))}
                        {trades.length === 0 && (
                            <tr><td colSpan="6" style={{ padding: "20px", textAlign: "center", color: "#666" }}>No trades found in Database.</td></tr>
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

// --- STYLES ---
const cardStyle = { background: "#1E1E1E", padding: "20px", borderRadius: "12px", border: "1px solid #333" };
const labelStyle = { color: "#888", fontSize: "14px", marginBottom: "5px" };
const valueStyle = { fontSize: "24px", fontWeight: "bold", color: "#fff" };
const thStyle = { padding: "15px", fontWeight: "600" };
const tdStyle = { padding: "15px", fontSize: "14px", color: "#eee" };

export default History;