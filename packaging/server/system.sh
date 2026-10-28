# Ubuntu and Red Hat differences, sourced by install-server.sh, uninstall-server.sh and the lanepulse command.
# Firewall: ufw (Ubuntu) or firewalld (Red Hat family). SELinux: Red Hat family, usually enforcing.

# Which firewall this server uses: ufw, firewalld or none.
fw_kind() {
  if command -v firewall-cmd >/dev/null 2>&1; then echo firewalld
  elif command -v ufw >/dev/null 2>&1; then echo ufw
  else echo none; fi
}

fw_active() {
  case "$(fw_kind)" in
    firewalld) firewall-cmd --state >/dev/null 2>&1;;
    ufw) ufw status 2>/dev/null | grep -q "Status: active";;
    *) return 1;;
  esac
}

# The firewalld rule for ALLOW (a network such as 192.168.1.0/24, or "any") and PORT.
_fw_rich() { echo "rule family=\"ipv4\" source address=\"$1\" port port=\"$2\" protocol=\"tcp\" accept"; }

# fw_open ALLOW PORT: let ALLOW reach PORT (kept across reboots).
fw_open() {
  case "$(fw_kind)" in
    firewalld)
      if [ "$1" = any ]; then firewall-cmd -q --permanent --add-port="$2/tcp"
      else firewall-cmd -q --permanent --add-rich-rule="$(_fw_rich "$1" "$2")"; fi
      firewall-cmd -q --reload;;
    ufw)
      if [ "$1" = any ]; then ufw allow "$2/tcp" >/dev/null
      else ufw allow from "$1" to any port "$2" proto tcp >/dev/null; fi;;
  esac
}

# fw_close ALLOW PORT: remove what fw_open added; nothing if it is not there.
fw_close() {
  case "$(fw_kind)" in
    firewalld)
      firewall-cmd --state >/dev/null 2>&1 || return 0
      if [ "$1" = any ]; then firewall-cmd -q --permanent --remove-port="$2/tcp" 2>/dev/null || true
      else firewall-cmd -q --permanent --remove-rich-rule="$(_fw_rich "$1" "$2")" 2>/dev/null || true; fi
      firewall-cmd -q --reload || true;;
    ufw)
      if [ "$1" = any ]; then ufw delete allow "$2/tcp" >/dev/null 2>&1 || true
      else ufw delete allow from "$1" to any port "$2" proto tcp >/dev/null 2>&1 || true; fi;;
  esac
}

# fw_enable ALLOW PORT: turn the firewall on with SSH and LanePulse allowed.
fw_enable() {
  case "$(fw_kind)" in
    firewalld)
      systemctl enable --now firewalld >/dev/null 2>&1
      firewall-cmd -q --permanent --add-service=ssh
      fw_open "$1" "$2"
      firewall-cmd --list-all;;
    ufw)
      ufw allow OpenSSH >/dev/null
      fw_open "$1" "$2"
      ufw --force enable >/dev/null
      ufw status;;
    *)
      echo "No firewall is installed. Ubuntu: sudo apt install ufw   Red Hat: sudo dnf install firewalld"
      return 1;;
  esac
}

fw_name() { case "$(fw_kind)" in firewalld) echo firewalld;; ufw) echo ufw;; *) echo "firewall";; esac; }

selinux_on() { command -v selinuxenabled >/dev/null 2>&1 && selinuxenabled; }

# selinux_label BASE: SELinux types for LanePulse's own Python (bin_t, so systemd may start it) and its logs
# (var_log_t, for systemd and logrotate). Recorded with semanage where it is installed, so a relabel keeps them.
selinux_label() {
  selinux_on || return 0
  local base="$1"
  if command -v semanage >/dev/null 2>&1; then
    semanage fcontext -a -t bin_t "$base/lanepulse-[^/]+/python/bin(/.*)?" 2>/dev/null \
      || semanage fcontext -m -t bin_t "$base/lanepulse-[^/]+/python/bin(/.*)?"
    semanage fcontext -a -t var_log_t "$base/logs(/.*)?" 2>/dev/null \
      || semanage fcontext -m -t var_log_t "$base/logs(/.*)?"
    restorecon -RF "$base"
  else
    restorecon -RF "$base"
    for d in "$base"/lanepulse-*/python/bin; do [ -d "$d" ] && chcon -R -t bin_t "$d"; done
    chcon -R -t var_log_t "$base/logs"
  fi
}

selinux_unlabel() {
  selinux_on && command -v semanage >/dev/null 2>&1 || return 0
  semanage fcontext -d "$1/lanepulse-[^/]+/python/bin(/.*)?" 2>/dev/null || true
  semanage fcontext -d "$1/logs(/.*)?" 2>/dev/null || true
}
