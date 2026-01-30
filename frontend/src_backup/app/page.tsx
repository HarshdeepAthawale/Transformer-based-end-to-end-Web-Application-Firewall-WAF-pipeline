'use client';

import React from 'react';
import { useWAFStore } from '@/lib/store';
import { MetricCard } from '@/components/dashboard/metric-card';
import { TrafficChart } from '@/components/dashboard/traffic-chart';
import { AttackTypeChart } from '@/components/dashboard/attack-type-chart';
import { RecentActivity } from '@/components/dashboard/recent-activity';
import {
  Activity,
  ShieldAlert,
  Zap,
  Clock,
  ShieldCheck
} from 'lucide-react';

export default function OverviewPage() {
  const overview = useWAFStore((state) => state.overview);

  if (!overview) return null;

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      <div className="flex flex-col gap-2">
        <h1 className="text-3xl font-bold tracking-tight text-zinc-950 dark:text-white">Dashboard Overview</h1>
        <p className="text-zinc-500 dark:text-zinc-400">Real-time security monitoring and threat intelligence summary.</p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="Total Requests"
          value={overview.totalRequests.toLocaleString()}
          trend={overview.requestTrend}
          icon={Activity}
          description="Total traffic processed in last 24h"
        />
        <MetricCard
          title="Blocked Attacks"
          value={overview.blockedRequests.toLocaleString()}
          trend={overview.blockedTrend}
          icon={ShieldAlert}
          description="Total malicious requests blocked"
        />
        <MetricCard
          title="Avg Latency"
          value={`${overview.avgLatency}ms`}
          trend={overview.latencyTrend}
          icon={Zap}
          description="Average response time per request"
        />
        <MetricCard
          title="Active Threats"
          value={overview.activeThreats}
          trend={overview.threatTrend}
          icon={ShieldCheck}
          description="Currently tracked high-risk IPs"
        />
      </div>

      <div className="grid gap-8 lg:grid-cols-5">
        <TrafficChart />
        <AttackTypeChart />
      </div>

      <div className="grid gap-8 lg:grid-cols-5">
        <RecentActivity />
        <div className="col-span-2 space-y-8">
          <div className="rounded-xl border border-zinc-200 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-950">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold text-zinc-950 dark:text-white uppercase tracking-wider">System Uptime</h3>
              <Clock className="h-4 w-4 text-zinc-400" />
            </div>
            <p className="text-2xl font-bold text-emerald-500">{overview.uptime}</p>
            <p className="mt-1 text-xs text-zinc-500">Last restart: 15 days ago</p>
          </div>

          <div className="rounded-xl border border-zinc-200 bg-indigo-600 p-6 text-white shadow-xl shadow-indigo-500/20">
            <h3 className="text-lg font-bold mb-2">AI Shield Active</h3>
            <p className="text-indigo-100 text-sm mb-4">The neural network is currently operating at 98.5% accuracy. No manual intervention required.</p>
            <button className="w-full py-2 bg-white text-indigo-600 rounded-lg text-sm font-bold hover:bg-indigo-50 transition-colors">
              View Model Status
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
