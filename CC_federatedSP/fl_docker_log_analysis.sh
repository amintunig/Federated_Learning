#!/bin/bash

##############################################
# Docker Logs Analysis for Federated Learning
# Analyzes logs by scenario for node performance
#
# Usage: ./fl_docker_log_analysis.sh
#
# Assumptions:
# - Containers are named or labeled with scenario info: scenario1, scenario2, scenario3, scenario4
# - Docker CLI is installed and user has permission to run docker commands
##############################################

# Array of scenario identifiers (adjust as per your naming)
SCENARIOS=("scenario1" "scenario2" "scenario3" "scenario4")

# Keywords to search for in logs
ERROR_KEYWORDS=("ERROR" "Failed" "Exception")
WARNING_KEYWORDS=("WARN" "Warning")

# Directory to store reports
REPORT_DIR="fl_docker_log_reports_$(date +%F)"
mkdir -p "$REPORT_DIR"

echo "Starting Docker log analysis for federated learning nodes..."

for scenario in "${SCENARIOS[@]}"; do
    echo "Processing logs for $scenario..."

    # Find containers matching the scenario name pattern
    CONTAINERS=$(docker ps --format '{{.Names}}' | grep "$scenario")

    if [ -z "$CONTAINERS" ]; then
        echo "No containers found for $scenario, skipping."
        continue
    fi

    # Temp file to aggregate logs for this scenario
    SCENARIO_LOG="$REPORT_DIR/${scenario}_combined.log"
    > "$SCENARIO_LOG"

    # Collect logs from each container for this scenario
    for container in $CONTAINERS; do
        echo "Collecting logs from container: $container"
        echo "===== Logs from $container =====" >> "$SCENARIO_LOG"
        # Collect last 500 lines - adjust as needed
        docker logs --tail 500 "$container" 2>&1 >> "$SCENARIO_LOG"
        echo "" >> "$SCENARIO_LOG"
    done

    # Analyze logs for errors and warnings
    TOTAL_LINES=$(wc -l < "$SCENARIO_LOG")
    ERROR_COUNT=0
    for kw in "${ERROR_KEYWORDS[@]}"; do
        ERROR_COUNT=$((ERROR_COUNT + $(grep -ci "$kw" "$SCENARIO_LOG")))
    done

    WARNING_COUNT=0
    for kw in "${WARNING_KEYWORDS[@]}"; do
        WARNING_COUNT=$((WARNING_COUNT + $(grep -ci "$kw" "$SCENARIO_LOG")))
    done

    # Extract top 5 error messages (case-insensitive)
    TOP_ERRORS=$(grep -i -E "$(IFS=\|; echo "${ERROR_KEYWORDS[*]}")" "$SCENARIO_LOG" | \
                 awk '{$1=$2=$3=""; print tolower($0)}' | \
                 sort | uniq -c | sort -nr | head -5)

    # Generate scenario report
    REPORT_FILE="$REPORT_DIR/${scenario}_report.txt"
    {
        echo "Federated Learning Node Log Analysis Report"
        echo "Scenario: $scenario"
        echo "Date: $(date)"
        echo "Containers analyzed: $(echo $CONTAINERS | wc -w)"
        echo "Total log lines processed: $TOTAL_LINES"
        echo "Total error count: $ERROR_COUNT"
        echo "Total warning count: $WARNING_COUNT"
        echo ""
        echo "Top 5 error messages:"
        echo "$TOP_ERRORS"
    } > "$REPORT_FILE"

    echo "Report generated for $scenario at $REPORT_FILE"
done

echo "Analysis complete. Reports are in the directory: $REPORT_DIR"
