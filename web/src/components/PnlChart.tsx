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
        borderColor: "#22c55e",
        backgroundColor: "rgba(34, 197, 94, 0.1)",
        fill: true,
        tension: 0.3,
        pointRadius: 2,
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: { display: true, position: "top" as const, labels: { color: "#9ca3af" } },
    },
    scales: {
      x: {
        title: { display: true, text: "Bet #", color: "#9ca3af" },
        ticks: { color: "#6b7280" },
        grid: { color: "#1f2937" },
      },
      y: {
        title: { display: true, text: "P&L ($)", color: "#9ca3af" },
        ticks: { color: "#6b7280" },
        grid: { color: "#1f2937" },
      },
    },
  };

  return <Line data={data} options={options} />;
}
