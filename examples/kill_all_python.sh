#!/usr/bin/env bash

ps x | grep python | awk  '{print $1}' | xargs kill