import { useCallback, useEffect, useState } from "react";
import {
  currentUserRequest,
  loginRequest,
  logoutRequest,
  refreshSession,
  registerRequest,
  subscribe,
} from "../api/authApi.js";

// React mirror of the in-memory auth session. The token itself is never
// persisted; a page reload intentionally lands on the sign-in screen.
export default function useAuth() {
  const [user, setUser] = useState(null);
  const [session, setSession] = useState({ token: null });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => subscribe((snapshot) => {
    setSession({ token: snapshot.token });
    if (!snapshot.token) setUser(null);
  }), []);

  const login = useCallback(async (username, password) => {
    setBusy(true);
    setError("");
    try {
      await loginRequest(username, password);
      const account = await currentUserRequest();
      setUser(account);
      return account;
    } catch (caught) {
      setError(caught.message || "Sign in failed. Please retry.");
      throw caught;
    } finally {
      setBusy(false);
    }
  }, []);

  const register = useCallback(async (username, password) => {
    setBusy(true);
    setError("");
    try {
      return await registerRequest(username, password);
    } catch (caught) {
      setError(caught.message || "Account creation failed. Please retry.");
      throw caught;
    } finally {
      setBusy(false);
    }
  }, []);

  const logout = useCallback(async () => {
    setBusy(true);
    try {
      await logoutRequest();
    } catch {
      // The local token is cleared even when the server could not confirm
      // revocation; the refresh cookie remains single-use and expires.
    } finally {
      setUser(null);
      setSession({ token: null });
      setBusy(false);
    }
  }, []);

  const refreshAccessToken = useCallback(async () => {
    try {
      const token = await refreshSession();
      return token;
    } catch (caught) {
      setError(caught.message || "Your session expired. Please sign in again.");
      setUser(null);
      throw caught;
    }
  }, []);

  return {
    user,
    accessToken: session.token,
    busy,
    error,
    clearError: () => setError(""),
    login,
    register,
    logout,
    refreshAccessToken,
  };
}
