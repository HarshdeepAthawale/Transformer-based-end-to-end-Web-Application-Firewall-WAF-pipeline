'use client'

import { useState, useEffect } from 'react'
import { Sidebar } from '@/components/sidebar'
import { Header } from '@/components/header'
import { ErrorBoundary } from '@/components/error-boundary'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { usersApi, User } from '@/lib/api'
import { Users, Plus, Search, AlertCircle, Shield } from 'lucide-react'

export default function UsersPage() {
  const [users, setUsers] = useState<User[]>([])
  const [currentUser, setCurrentUser] = useState<User | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [showAddDialog, setShowAddDialog] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [newUsername, setNewUsername] = useState('')
  const [newEmail, setNewEmail] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [newRole, setNewRole] = useState('viewer')
  const [newFullName, setNewFullName] = useState('')

  useEffect(() => {
    fetchUsers()
    fetchCurrentUser()
  }, [])

  const fetchUsers = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await usersApi.getUsers()
      if (response.success) setUsers(response.data)
    } catch (err: any) {
      if (!err?.isNetworkError) {
        setError(err?.message || 'Failed to fetch users')
      }
    } finally {
      setLoading(false)
    }
  }

  const fetchCurrentUser = async () => {
    try {
      const response = await usersApi.getCurrentUser()
      if (response.success) setCurrentUser(response.data)
    } catch (err: any) {
      // Silently handle network errors
    }
  }

  const handleAdd = async () => {
    if (!newUsername.trim() || !newEmail.trim() || !newPassword.trim()) return

    setLoading(true)
    setError(null)
    try {
      await usersApi.createUser({
        username: newUsername,
        email: newEmail,
        password: newPassword,
        role: newRole,
        full_name: newFullName || undefined,
      })
      setNewUsername('')
      setNewEmail('')
      setNewPassword('')
      setNewRole('viewer')
      setNewFullName('')
      setShowAddDialog(false)
      fetchUsers()
    } catch (err: any) {
      if (!err?.isNetworkError) {
        setError(err?.message || 'Failed to create user')
      }
    } finally {
      setLoading(false)
    }
  }

  const filteredUsers = users.filter(user =>
    user.username.toLowerCase().includes(searchQuery.toLowerCase()) ||
    user.email.toLowerCase().includes(searchQuery.toLowerCase()) ||
    user.full_name?.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const roleColors: Record<string, string> = {
    admin: 'destructive',
    operator: 'default',
    viewer: 'secondary',
  }

  return (
    <div className="flex h-screen bg-background">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <main className="flex-1 overflow-y-auto p-6">
          <ErrorBoundary>
            <div className="max-w-7xl mx-auto space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <h1 className="text-3xl font-bold">User Management</h1>
                  <p className="text-muted-foreground mt-1">Manage system users and permissions</p>
                </div>
                <Button onClick={() => setShowAddDialog(true)}>
                  <Plus className="mr-2 h-4 w-4" />
                  Add User
                </Button>
              </div>

              {error && (
                <Card className="p-4 bg-destructive/10 border-destructive">
                  <div className="flex items-center gap-2">
                    <AlertCircle className="h-4 w-4 text-destructive" />
                    <p className="text-sm text-destructive">{error}</p>
                  </div>
                </Card>
              )}

              {currentUser && (
                <Card className="p-4 bg-muted">
                  <div className="flex items-center gap-2">
                    <Shield className="h-4 w-4" />
                    <span className="text-sm">
                      Logged in as: <strong>{currentUser.username}</strong> ({currentUser.role})
                    </span>
                  </div>
                </Card>
              )}

              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <Card className="p-4">
                  <div className="text-sm text-muted-foreground">Total Users</div>
                  <div className="text-2xl font-bold mt-1">{users.length}</div>
                </Card>
                <Card className="p-4">
                  <div className="text-sm text-muted-foreground">Admins</div>
                  <div className="text-2xl font-bold mt-1">
                    {users.filter(u => u.role === 'admin').length}
                  </div>
                </Card>
                <Card className="p-4">
                  <div className="text-sm text-muted-foreground">Active</div>
                  <div className="text-2xl font-bold mt-1">
                    {users.filter(u => u.is_active).length}
                  </div>
                </Card>
                <Card className="p-4">
                  <div className="text-sm text-muted-foreground">Viewers</div>
                  <div className="text-2xl font-bold mt-1">
                    {users.filter(u => u.role === 'viewer').length}
                  </div>
                </Card>
              </div>

              <Card>
                <div className="p-4 border-b">
                  <div className="flex items-center gap-4">
                    <div className="flex-1 relative">
                      <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                      <Input
                        placeholder="Search users..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="pl-10"
                      />
                    </div>
                  </div>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead className="bg-muted">
                      <tr>
                        <th className="px-4 py-3 text-left text-sm font-medium">Username</th>
                        <th className="px-4 py-3 text-left text-sm font-medium">Email</th>
                        <th className="px-4 py-3 text-left text-sm font-medium">Full Name</th>
                        <th className="px-4 py-3 text-left text-sm font-medium">Role</th>
                        <th className="px-4 py-3 text-left text-sm font-medium">Status</th>
                        <th className="px-4 py-3 text-left text-sm font-medium">Last Login</th>
                        <th className="px-4 py-3 text-left text-sm font-medium">Created</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredUsers.map((user) => (
                        <tr key={user.id} className="border-b hover:bg-muted/50">
                          <td className="px-4 py-3 font-medium">{user.username}</td>
                          <td className="px-4 py-3">{user.email}</td>
                          <td className="px-4 py-3">{user.full_name || '-'}</td>
                          <td className="px-4 py-3">
                            <Badge variant={roleColors[user.role] as any}>
                              {user.role}
                            </Badge>
                          </td>
                          <td className="px-4 py-3">
                            <Badge variant={user.is_active ? 'default' : 'secondary'}>
                              {user.is_active ? 'Active' : 'Inactive'}
                            </Badge>
                          </td>
                          <td className="px-4 py-3 text-sm text-muted-foreground">
                            {user.last_login ? new Date(user.last_login).toLocaleString() : 'Never'}
                          </td>
                          <td className="px-4 py-3 text-sm text-muted-foreground">
                            {new Date(user.created_at).toLocaleDateString()}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {filteredUsers.length === 0 && (
                    <div className="p-8 text-center text-muted-foreground">No users found</div>
                  )}
                </div>
              </Card>
            </div>

            <Dialog open={showAddDialog} onOpenChange={setShowAddDialog}>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Create User</DialogTitle>
                  <DialogDescription>
                    Create a new user account with specified role and permissions
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-4">
                  <div>
                    <label className="text-sm font-medium">Username</label>
                    <Input
                      value={newUsername}
                      onChange={(e) => setNewUsername(e.target.value)}
                      placeholder="johndoe"
                    />
                  </div>
                  <div>
                    <label className="text-sm font-medium">Email</label>
                    <Input
                      type="email"
                      value={newEmail}
                      onChange={(e) => setNewEmail(e.target.value)}
                      placeholder="john@example.com"
                    />
                  </div>
                  <div>
                    <label className="text-sm font-medium">Password</label>
                    <Input
                      type="password"
                      value={newPassword}
                      onChange={(e) => setNewPassword(e.target.value)}
                      placeholder="••••••••"
                    />
                  </div>
                  <div>
                    <label className="text-sm font-medium">Full Name (optional)</label>
                    <Input
                      value={newFullName}
                      onChange={(e) => setNewFullName(e.target.value)}
                      placeholder="John Doe"
                    />
                  </div>
                  <div>
                    <label className="text-sm font-medium">Role</label>
                    <Select value={newRole} onValueChange={setNewRole}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="viewer">Viewer</SelectItem>
                        <SelectItem value="operator">Operator</SelectItem>
                        <SelectItem value="admin">Admin</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="flex justify-end gap-2">
                    <Button variant="outline" onClick={() => setShowAddDialog(false)}>
                      Cancel
                    </Button>
                    <Button onClick={handleAdd} disabled={loading || !newUsername.trim() || !newEmail.trim() || !newPassword.trim()}>
                      Create User
                    </Button>
                  </div>
                </div>
              </DialogContent>
            </Dialog>
          </ErrorBoundary>
        </main>
      </div>
    </div>
  )
}
