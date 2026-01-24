'use client'

import { Sidebar } from '@/components/sidebar'
import { Header } from '@/components/header'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { useState } from 'react'
import { Bell, Lock, Users, Zap } from 'lucide-react'

export default function SettingsPage() {
  const [timeRange, setTimeRange] = useState('24h')
  const [notifications, setNotifications] = useState(true)
  const [emailAlerts, setEmailAlerts] = useState(true)
  const [autoBlock, setAutoBlock] = useState(true)

  return (
    <div className="flex h-screen bg-background text-foreground">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header timeRange={timeRange} onTimeRangeChange={setTimeRange} />
        <main className="flex-1 overflow-auto">
          <div className="p-6 space-y-6">
            <div>
              <h2 className="text-2xl font-bold mb-2">Settings</h2>
              <p className="text-muted-foreground">Manage dashboard settings and preferences</p>
            </div>

            <div className="space-y-6 max-w-2xl">
              <Card className="p-6">
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-4">
                    <div className="p-3 bg-black/20 rounded-lg mt-1">
                      <Bell className="w-5 h-5 text-black" />
                    </div>
                    <div>
                      <h3 className="font-semibold">Notifications</h3>
                      <p className="text-sm text-muted-foreground">Receive alerts and updates</p>
                    </div>
                  </div>
                  <button
                    onClick={() => setNotifications(!notifications)}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                      notifications ? 'bg-black' : 'bg-gray-700'
                    }`}
                  >
                    <span
                      className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                        notifications ? 'translate-x-6' : 'translate-x-1'
                      }`}
                    />
                  </button>
                </div>
              </Card>

              <Card className="p-6">
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-4">
                    <div className="p-3 bg-orange-500/20 rounded-lg mt-1">
                      <Zap className="w-5 h-5 text-orange-400" />
                    </div>
                    <div>
                      <h3 className="font-semibold">Auto Block Threats</h3>
                      <p className="text-sm text-muted-foreground">Automatically block detected threats</p>
                    </div>
                  </div>
                  <button
                    onClick={() => setAutoBlock(!autoBlock)}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                      autoBlock ? 'bg-orange-500' : 'bg-gray-700'
                    }`}
                  >
                    <span
                      className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                        autoBlock ? 'translate-x-6' : 'translate-x-1'
                      }`}
                    />
                  </button>
                </div>
              </Card>

              <Card className="p-6">
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-4">
                    <div className="p-3 bg-green-500/20 rounded-lg mt-1">
                      <Users className="w-5 h-5 text-green-400" />
                    </div>
                    <div>
                      <h3 className="font-semibold">Email Alerts</h3>
                      <p className="text-sm text-muted-foreground">Send critical alerts via email</p>
                    </div>
                  </div>
                  <button
                    onClick={() => setEmailAlerts(!emailAlerts)}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                      emailAlerts ? 'bg-green-500' : 'bg-gray-700'
                    }`}
                  >
                    <span
                      className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                        emailAlerts ? 'translate-x-6' : 'translate-x-1'
                      }`}
                    />
                  </button>
                </div>
              </Card>

              <Card className="p-6">
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-4">
                    <div className="p-3 bg-red-500/20 rounded-lg mt-1">
                      <Lock className="w-5 h-5 text-red-400" />
                    </div>
                    <div>
                      <h3 className="font-semibold">API Key Management</h3>
                      <p className="text-sm text-muted-foreground">Manage your API keys and access tokens</p>
                    </div>
                  </div>
                  <Button className="bg-accent hover:bg-accent/90">View Keys</Button>
                </div>
              </Card>

              <div className="pt-4 space-y-3">
                <Button className="w-full bg-primary hover:bg-primary/90">Save Settings</Button>
                <Button variant="outline" className="w-full bg-transparent">Cancel</Button>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}
