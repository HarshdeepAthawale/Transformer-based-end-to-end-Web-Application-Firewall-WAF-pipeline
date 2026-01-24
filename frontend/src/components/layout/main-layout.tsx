'use client';

import React from 'react';
import { Sidebar } from './sidebar';
import { Header } from './header';
import { DashboardProvider } from '../providers/dashboard-provider';

export function MainLayout({ children }: { children: React.ReactNode }) {
  return (
    <DashboardProvider>
      <div className="flex min-h-screen bg-zinc-50 dark:bg-black">
        <Sidebar />
        <div className="flex flex-1 flex-col md:pl-64">
          <Header />
          <main className="flex-1 p-8">
            {children}
          </main>
        </div>
      </div>
    </DashboardProvider>
  );
}
