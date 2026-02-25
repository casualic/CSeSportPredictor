"use client";

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from "chart.js";
import { Line } from "react-chartjs-2";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler);

interface PnlChartProps {
  cumulative: number[];
}

export default function PnlChart({ cumulative }: PnlChartProps) {
  const data = {
    labels: cumulative.map((_, i) => i + 1),
    datasets: [
      {
        label: "Cumulative P&L ($)",
        data: cumulative,
        borderColor: "#2563eb",
        backgroundColor: "rgba(37, 99, 235, 0.06)",
        fill: true,
        tension: 0.3,
        pointRadius: 2,
        pointBackgroundColor: "#2563eb",
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        display: true,
        position: "top" as const,
        labels: { color: "#475569", font: { size: 12 } },
      },
    },
    scales: {
      x: {
        title: { display: true, text: "Bet #", color: "#94a3b8" },
        ticks: { color: "#94a3b8" },
        grid: { color: "#f1f5f9" },
      },
      y: {
        title: { display: true, text: "P&L ($)", color: "#94a3b8" },
        ticks: { color: "#94a3b8" },
        grid: { color: "#f1f5f9" },
      },
    },
  };

  return <Line data={data} options={options} />;
}
