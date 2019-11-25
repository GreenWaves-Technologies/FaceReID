#!/bin/sh

set -e -x

export DOCKER_BUILDKIT=1
docker build --ssh default -t registry.gitlab.com/xperience-ai/gap8-reid-demo:v2 .
docker push registry.gitlab.com/xperience-ai/gap8-reid-demo:v2
