#pragma once

#include "Modules/ModuleManager.h"

class FPedestrianRuntimeModule : public IModuleInterface
{
public:
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;
};
