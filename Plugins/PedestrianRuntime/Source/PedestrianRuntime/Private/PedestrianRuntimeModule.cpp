#include "PedestrianRuntimeModule.h"

#include "PedestrianRuntimeLog.h"

#define LOCTEXT_NAMESPACE "FPedestrianRuntimeModule"

void FPedestrianRuntimeModule::StartupModule()
{
	UE_LOG(LogPedestrianRuntime, Log, TEXT("PedestrianRuntime module started."));
}

void FPedestrianRuntimeModule::ShutdownModule()
{
	UE_LOG(LogPedestrianRuntime, Log, TEXT("PedestrianRuntime module stopped."));
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FPedestrianRuntimeModule, PedestrianRuntime)
