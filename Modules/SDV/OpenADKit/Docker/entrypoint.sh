#!/usr/bin/env bash
# cspell:ignore openbox, VNC, tigervnc, novnc, websockify, newkey, xstartup, pixelformat, vncserver, autoconnect, vncpasswd
set -e

# Configuration with defaults
: "${VNC_GEOMETRY:=1280x720}"
: "${VNC_DEPTH:=24}"
: "${VNC_PASSWORD:=visualizer}"
: "${GUI_APP:=}"
: "${DISPLAY:=:99}"

export DISPLAY

log() {
    echo -e "\033[32m[visualizer]\033[0m $1"
}

warn() {
    echo -e "\033[33m[visualizer]\033[0m $1"
}

error() {
    echo -e "\033[31m[visualizer]\033[0m $1"
}

configure_openbox() {
    mkdir -p /etc/xdg/openbox

    # Openbox configuration - maximize windows by default
    cat >/etc/xdg/openbox/rc.xml <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<openbox_config xmlns="http://openbox.org/3.4/rc"
                xmlns:xi="http://www.w3.org/2001/XInclude">
  <resistance>
    <strength>10</strength>
    <screen_edge_strength>20</screen_edge_strength>
  </resistance>
  <focus>
    <focusNew>yes</focusNew>
    <followMouse>no</followMouse>
  </focus>
  <placement>
    <policy>Smart</policy>
    <center>yes</center>
  </placement>
  <theme>
    <name>Clearlooks</name>
  </theme>
  <desktops>
    <number>1</number>
  </desktops>
  <applications>
    <application class="*">
      <maximized>yes</maximized>
      <focus>yes</focus>
    </application>
  </applications>
</openbox_config>
EOF

    # Autostart configuration
    cat >/etc/xdg/openbox/autostart <<EOF
# Log startup
echo "Openbox autostart at \$(date)" >> /tmp/autostart.log

# Launch GUI application if specified
if [ -n "$GUI_APP" ]; then
    echo "Launching: $GUI_APP" >> /tmp/autostart.log
    $GUI_APP &
fi
EOF
}

start_vnc() {
    log "Setting up VNC password..."
    mkdir -p ~/.vnc
    echo "$VNC_PASSWORD" | vncpasswd -f > ~/.vnc/passwd
    chmod 600 ~/.vnc/passwd

    log "Starting VNC server (geometry: $VNC_GEOMETRY, depth: $VNC_DEPTH)..."
    vncserver $DISPLAY \
        -geometry "$VNC_GEOMETRY" \
        -depth "$VNC_DEPTH" \
        -localhost no \
        -SecurityTypes VncAuth

    if [ $? -ne 0 ]; then
        error "Failed to start VNC server"
        exit 1
    fi
    log "VNC server started on display $DISPLAY"
}

start_novnc() {
    log "Starting NoVNC web server..."
    websockify \
        --daemon \
        --web=/usr/share/novnc/ \
        --cert=/etc/ssl/certs/novnc.crt \
        --key=/etc/ssl/private/novnc.key \
        6080 \
        localhost:5999

    if [ $? -ne 0 ]; then
        error "Failed to start NoVNC"
        exit 1
    fi
    log "NoVNC started on port 6080"
}

print_access_info() {
    local LOCAL_IP
    LOCAL_IP=$(hostname -I 2>/dev/null | cut -d' ' -f1 || echo "localhost")
    
    echo ""
    echo -e "\033[32m═══════════════════════════════════════════════════════════════════════\033[0m"
    echo -e "\033[32m  GUI Visualizer Ready!\033[0m"
    echo -e "\033[32m═══════════════════════════════════════════════════════════════════════\033[0m"
    echo ""
    echo -e "  \033[36mBrowser Access (NoVNC):\033[0m"
    echo -e "    http://${LOCAL_IP}:6080/vnc.html?resize=scale&autoconnect=true&password=${VNC_PASSWORD}"
    echo ""
    echo -e "  \033[36mVNC Client Access:\033[0m"
    echo -e "    ${LOCAL_IP}:5999"
    echo ""
    echo -e "  \033[36mPassword:\033[0m ${VNC_PASSWORD}"
    echo ""
    if [ -n "$GUI_APP" ]; then
        echo -e "  \033[36mGUI Application:\033[0m ${GUI_APP}"
    else
        echo -e "  \033[33mNo GUI_APP specified - empty desktop\033[0m"
        echo -e "  \033[33mSet GUI_APP env var to auto-launch an application\033[0m"
    fi
    echo ""
    echo -e "\033[32m═══════════════════════════════════════════════════════════════════════\033[0m"
    echo ""
}

main() {
    log "Initializing GUI Visualizer..."
    
    # Configure openbox window manager
    configure_openbox
    
    # Start VNC server
    start_vnc
    
    # Give VNC a moment to initialize
    sleep 2
    
    # Start NoVNC for browser access
    start_novnc
    
    # Print access information
    print_access_info
    
    # Execute any passed command, or keep container running
    if [ $# -gt 0 ]; then
        log "Executing command: $*"
        exec "$@"
    else
        log "Container running. Press Ctrl+C to stop."
        sleep infinity
    fi
}

main "$@"
