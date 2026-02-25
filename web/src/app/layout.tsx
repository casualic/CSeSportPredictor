import type { Metadata } from "next";
import { Source_Serif_4, IBM_Plex_Sans, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";
import Navbar from "@/components/Navbar";

const sourceSerif = Source_Serif_4({
  variable: "--font-source-serif",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
});

const ibmPlexSans = IBM_Plex_Sans({
  variable: "--font-ibm-plex-sans",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600"],
});

const ibmPlexMono = IBM_Plex_Mono({
  variable: "--font-ibm-plex-mono",
  subsets: ["latin"],
  weight: ["400", "500"],
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
    <html lang="en">
      <body
        className={`${sourceSerif.variable} ${ibmPlexSans.variable} ${ibmPlexMono.variable} antialiased bg-slate-50 text-slate-900`}
      >
        <Navbar />
        <main className="max-w-[1120px] mx-auto px-7 py-8">{children}</main>
        <footer className="max-w-[1120px] mx-auto px-7 py-[18px] mt-9 border-t border-slate-200 flex justify-between text-xs text-slate-400">
          <span>CS2 Predictor &mdash; FSVM + XGB Ensemble</span>
          <span>Last updated: {new Date().toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}</span>
        </footer>
      </body>
    </html>
  );
}
