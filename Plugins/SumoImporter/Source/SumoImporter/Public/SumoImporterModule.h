#pragma once

#include "Modules/ModuleManager.h"

class FSumoImporterModule : public IModuleInterface
{
public:
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;
};
