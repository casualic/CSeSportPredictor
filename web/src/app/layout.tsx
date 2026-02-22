import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import Navbar from "@/components/Navbar";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "CS2 Predictor",
  description: "CS2 match predictions with FSVM + XGB ensemble",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased bg-gray-950 text-gray-200`}>
        <Navbar />
        <main className="max-w-6xl mx-auto px-4 py-6">{children}</main>
        <footer className="border-t border-gray-800 py-4 mt-8">
          <div className="text-center text-gray-500 text-sm">
            CS2 Match Predictor &mdash; FSVM + XGB Ensemble
          </div>
        </footer>
      </body>
    </html>
  );
}
